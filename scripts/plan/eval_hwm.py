"""Evaluate a Hierarchical World Model (HWM) using two-level MPC on a dataset.

Architecture
------------
L2 CEM : searches in latent macro-action space  (HWMCostModel wraps HWM PreJEPA)
L1 CEM : plans primitive actions toward each subgoal embedding
         (_FixedGoalCostModel wraps low-level PreJEPA)
"""

import os

os.environ['MUJOCO_GL'] = 'egl'

import json
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torch import nn
from torchvision.transforms import v2 as transforms
from transformers import AutoModel


# ── Model loading ─────────────────────────────────────────────────────────────

def _build_and_load_prejepa(run_name: str, cache_dir: str | None = None) -> swm.wm.PreJEPA:
    """Build a PreJEPA from config.json and load weights from the Lightning checkpoint.

    Both training scripts (prejepa.py, hwm.py) save:
      - ``<run_dir>/config.json``        — architecture hyper-parameters
      - ``<run_dir>/<run_name>_weights.ckpt`` — Lightning checkpoint (state_dict under 'model.' prefix)

    Args:
        run_name: Folder name under the checkpoints directory.
        cache_dir: Override for the checkpoint root.

    Returns:
        PreJEPA in eval mode with frozen weights.
    """
    ckpt_root = Path(cache_dir or swm.data.utils.get_cache_dir(sub_folder='checkpoints'))
    run_dir = ckpt_root / run_name

    cfg = json.loads((run_dir / 'config.json').read_text())
    ckpt_path = ckpt_root / f'{run_name}_weights.ckpt'
    if not ckpt_path.exists():
        # Fall back to the single .pt file if no Lightning ckpt
        pt_files = list(run_dir.glob('*.pt'))
        if len(pt_files) != 1:
            raise FileNotFoundError(
                f'Expected exactly one .pt or one _weights.ckpt for {run_name}; '
                f'found {[str(p) for p in pt_files]}'
            )
        ckpt_path = pt_files[0]
        use_lightning = False
    else:
        use_lightning = True

    # Build backbone
    backbone = AutoModel.from_pretrained(cfg['backbone']['name'])
    embed_dim = backbone.config.hidden_size        # 384 for DINOv2-small
    num_patches = (cfg['image_size'] // cfg['patch_size']) ** 2  # 256

    # Extra encoders: key → Embedder(in_chans=raw_dim, emb_dim=emb_dim)
    encoding: dict[str, int] = cfg['wm']['encoding']   # key → emb_dim
    extra_dims: dict[str, int] = cfg.get('extra_dims', {})   # key → raw_dim

    extra_encoders = nn.ModuleDict()
    for key, emb_dim in encoding.items():
        in_chans = extra_dims.get(key, emb_dim)   # fall back to emb_dim if raw_dim unknown
        extra_encoders[key] = swm.wm.prejepa.Embedder(in_chans=in_chans, emb_dim=emb_dim)

    # Predictor
    dim = embed_dim + sum(encoding.values())
    pred_cfg = cfg['predictor']
    predictor = swm.wm.prejepa.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg['wm']['history_size'],
        dim=dim,
        depth=pred_cfg['depth'],
        heads=pred_cfg['heads'],
        mlp_dim=pred_cfg['mlp_dim'],
        dim_head=pred_cfg['dim_head'],
        dropout=pred_cfg['dropout'],
        emb_dropout=pred_cfg['emb_dropout'],
    )

    model = swm.wm.PreJEPA(
        encoder=spt.backbone.EvalOnly(backbone),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg['wm']['history_size'],
        num_pred=cfg['wm']['num_preds'],
        interpolate_pos_encoding=True,
    )

    # Load weights
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if use_lightning and 'state_dict' in raw:
        state_dict = {
            k[len('model.'):]: v
            for k, v in raw['state_dict'].items()
            if k.startswith('model.')
        }
    else:
        state_dict = raw   # plain state dict saved by save_pretrained

    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model


# ── Transform / dataset helpers ───────────────────────────────────────────────

def img_transform(cfg, dtype=torch.float32):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_episodes_length(dataset, episodes):
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    return np.array([np.max(step_idx[episode_idx == ep]) + 1 for ep in episodes])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path='./config', config_name='pusht_hwm')
def run(cfg: DictConfig):
    """Run HWM evaluation: L2 CEM (macro-action) + L1 CEM (primitive actions)."""
    l1_steps_per_subgoal = cfg.l1.horizon * cfg.low_level.frameskip
    assert l1_steps_per_subgoal * cfg.l2.receding_horizon <= cfg.eval.eval_budget, (
        f'L1 steps ({l1_steps_per_subgoal}) × l2_receding ({cfg.l2.receding_horizon}) '
        f'must be ≤ eval_budget ({cfg.eval.eval_budget})'
    )

    # ── Environment ──────────────────────────────────────────────────────────
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # ── Image transform ───────────────────────────────────────────────────────
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    # ── Dataset / normalizers ─────────────────────────────────────────────────
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # ── Policy ────────────────────────────────────────────────────────────────
    policy_name = cfg.get('policy', 'random')

    if policy_name != 'random':
        # Load HWM (L2) and low-level WM (L1)
        hwm = _build_and_load_prejepa(cfg.policy, cfg.get('cache_dir'))
        low_level = _build_and_load_prejepa(cfg.low_level.checkpoint, cfg.get('cache_dir'))

        hwm = hwm.to('cuda')
        low_level = low_level.to('cuda')
        if cfg.get('bf16', False):
            hwm = hwm.to(torch.bfloat16)
            low_level = low_level.to(torch.bfloat16)

        # ── L2 solver: macro-action CEM ───────────────────────────────────────
        # HWMCostModel bypasses PreJEPA.rollout (which hardcodes 'action' key)
        # and embeds the latent_action candidates directly.
        l2_config = swm.PlanConfig(
            horizon=cfg.l2.horizon,
            receding_horizon=cfg.l2.receding_horizon,
            action_block=1,           # each candidate step is 1 macro latent vector
        )
        hwm_cost = swm.policy.HWMCostModel(
            hwm, action_key=cfg.l2.get('action_key', 'latent_action')
        )
        l2_solver = hydra.utils.instantiate(cfg.l2.solver, model=hwm_cost)

        # ── L1 solver: primitive-action CEM ──────────────────────────────────
        # HWMPolicy._replan swaps in a _FixedGoalCostModel per subgoal.
        # We pass low_level as a placeholder model here.
        l1_config = swm.PlanConfig(
            horizon=cfg.l1.horizon,
            receding_horizon=cfg.l1.horizon,   # execute the full plan to reach each subgoal
            action_block=cfg.low_level.frameskip,
        )
        l1_solver = hydra.utils.instantiate(cfg.l1.solver, model=low_level)

        policy = swm.policy.HWMPolicy(
            hwm_model=hwm,
            low_level_model=low_level,
            l2_solver=l2_solver,
            l1_solver=l1_solver,
            l2_config=l2_config,
            l1_config=l1_config,
            macro_action_dim=cfg.macro_action.latent_dim,
            process=process,
            transform=transform,
        )

    else:
        policy = swm.policy.RandomPolicy()

    # ── Output path ───────────────────────────────────────────────────────────
    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if policy_name != 'random'
        else Path(__file__).parent
    )

    # ── Sample evaluation episodes ────────────────────────────────────────────
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([
        max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)
    ])
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    chosen = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    chosen = np.sort(valid_indices[chosen])

    eval_episodes = dataset.get_row_data(chosen)[col_name]
    eval_start_idx = dataset.get_row_data(chosen)['step_idx']

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError('Not enough episodes with sufficient length for evaluation.')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    world.set_policy(policy)

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get('callables'), resolve=True),
            video=results_path,
        )
    end_time = time.time()

    print(metrics)

    out_file = results_path / cfg.output.filename
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open('a') as f:
        f.write('\n')
        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
