import pytest
import torch
import numpy as np
from copy import deepcopy
from types import SimpleNamespace
from gymnasium.spaces import Box
from omegaconf import OmegaConf

from stable_worldmodel.wm.tdmpc2 import TDMPC2, tdmpc2_forward


class _Config(dict):
    __getattr__ = dict.__getitem__


class _ForwardContext:
    def __init__(self, model):
        self.model = model

    def log_dict(self, *_args, **_kwargs):
        pass


def _make_config():
    wm = _Config(
        encoding={'observation': 8},
        horizon=3,
        mlp_dim=16,
        enc_dim=16,
        simnorm_dim=8,
        num_q=2,
        rho=0.5,
        tau=0.01,
        consistency_coef=20.0,
        reward_coef=0.1,
        value_coef=0.1,
        discount=0.99,
        entropy_coef=1e-4,
        num_bins=11,
        vmin=-6,
        vmax=2,
    )
    return _Config(
        action_dim=2,
        extra_dims={'observation': 3},
        wm=wm,
    )


def _make_batch(cfg):
    generator = torch.Generator().manual_seed(7)
    batch_size = 8
    num_steps = cfg.wm.horizon + 1
    return {
        'observation': torch.randn(
            batch_size, num_steps, 3, generator=generator
        ),
        'action': torch.randn(
            batch_size,
            num_steps,
            cfg.action_dim,
            generator=generator,
        ).tanh(),
        'reward': torch.randn(batch_size, num_steps, generator=generator),
    }


def _target_q_state(model):
    return torch.cat(
        [p.detach().flatten() for p in model.target_qs.parameters()]
    ).clone()


def test_validation_forward_keeps_running_scale_frozen():
    cfg = _make_config()
    model = TDMPC2(cfg)
    scale_before = model.scale.value.clone()
    target_q_before = _target_q_state(model)

    output = tdmpc2_forward(
        _ForwardContext(model),
        _make_batch(cfg),
        stage='validate',
        cfg=cfg,
    )

    assert torch.equal(model.scale.value, scale_before)
    assert torch.equal(_target_q_state(model), target_q_before)
    assert torch.isfinite(output['loss'])


# 'fit' is what stable_pretraining.Module.training_step passes; 'train' is
# what the online loop in scripts/expert/tdmpc2_online.py passes.
@pytest.mark.parametrize('stage', ['train', 'fit'])
def test_training_forward_updates_running_scale(stage):
    cfg = _make_config()
    model = TDMPC2(cfg)
    scale_before = model.scale.value.clone()
    target_q_before = _target_q_state(model)

    tdmpc2_forward(
        _ForwardContext(model),
        _make_batch(cfg),
        stage=stage,
        cfg=cfg,
    )

    assert not torch.equal(model.scale.value, scale_before)
    assert not torch.equal(_target_q_state(model), target_q_before)


def _offline_config(goal_conditioned=True):
    cfg = _make_config()
    cfg = OmegaConf.create({**cfg, 'wm': dict(cfg.wm)})
    cfg.extra_dims.observation = 6 if goal_conditioned else 3
    cfg.preprocessing = {
        'goal_obs_key': 'observation' if goal_conditioned else None,
        'statistics': {
            'observation': {
                'mean': [-2.0, 1.0, 3.0, 5.0, -1.0, 8.0]
                if goal_conditioned
                else [-2.0, 1.0, 3.0],
                'std': [2.0, 4.0, 0.5, 3.0, 1.5, 2.0]
                if goal_conditioned
                else [2.0, 4.0, 0.5],
            }
        },
    }
    return cfg


def _planning_inputs(cfg, leading=(2,), raw=False):
    generator = torch.Generator().manual_seed(5)
    obs = torch.randn(*leading, 3, generator=generator)
    info = {'observation': obs}
    if cfg.preprocessing.goal_obs_key is not None:
        goal = torch.randn(*leading, 3, generator=generator) + 7
        info['goal_observation'] = goal
        obs = torch.cat([obs, goal], dim=-1)
    stats = cfg.preprocessing.statistics.observation
    expected = (obs - torch.tensor(stats.mean)) / torch.tensor(stats.std)
    if raw:
        return info, {'observation': expected}
    if cfg.preprocessing.goal_obs_key is not None:
        info = dict(zip(info, expected.split(3, dim=-1)))
    else:
        info = {'observation': expected}
    return info, {'observation': expected}


@pytest.mark.parametrize('goal_conditioned', [False, True])
@pytest.mark.parametrize('leading', [(2,), (2, 4)])
@pytest.mark.parametrize('with_time_dim', [False, True])
def test_planning_cost_matches_training_coordinates(
    goal_conditioned, leading, with_time_dim
):
    cfg = _offline_config(goal_conditioned)
    model = TDMPC2(cfg).eval()
    reference = deepcopy(model)
    del reference.cfg.preprocessing
    info, prepared = _planning_inputs(cfg, leading)
    if with_time_dim:
        info = {key: value.unsqueeze(-2) for key, value in info.items()}
    original = {key: value.clone() for key, value in info.items()}
    actions = torch.linspace(-0.8, 0.8, 2 * 4 * 3 * 2).reshape(2, 4, 3, 2)

    torch.testing.assert_close(
        model.get_cost(info, actions),
        reference.get_cost(prepared, actions),
        rtol=0,
        atol=0,
    )
    for key in info:
        torch.testing.assert_close(info[key], original[key], rtol=0, atol=0)


@pytest.mark.parametrize('goal_conditioned', [False, True])
@pytest.mark.parametrize('with_time_dim', [False, True])
def test_actor_prefix_uses_training_coordinates(
    goal_conditioned, with_time_dim
):
    cfg = _offline_config(goal_conditioned)
    model = TDMPC2(cfg).eval()
    reference = deepcopy(model)
    del reference.cfg.preprocessing
    info, prepared = _planning_inputs(cfg)
    if with_time_dim:
        info = {key: value.unsqueeze(1) for key, value in info.items()}
    prefix = torch.full((2, 1, 2), 0.2)

    torch.manual_seed(19)
    expected = reference.get_action(prepared, horizon=3, prefix_actions=prefix)
    torch.manual_seed(19)
    actual = model.get_action(info, horizon=3, prefix_actions=prefix)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _eval_config():
    return OmegaConf.create(
        {
            'dataset': {'keys_to_cache': ['observation', 'action']},
            'eval': {'img_size': 64},
            'solver': {
                '_target_': 'stable_worldmodel.planning.CEMSolver',
                'num_samples': 8,
                'topk': 2,
                'n_steps': 2,
                'var_scale': 0.01,
                'seed': 17,
            },
            'objective': {'_target_': 'stable_worldmodel.planning.GoalMSE'},
            'plan_config': {'horizon': 3, 'receding_horizon': 1},
        }
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('with_time_dim', [False, True])
def test_checkpoint_eval_matches_reference_without_refitting(
    tmp_path, dtype, with_time_dim
):
    pytest.importorskip('stable_pretraining')
    from scripts.plan.eval_wm import make_model_policy
    from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
    from stable_worldmodel.planning import CEMSolver
    from stable_worldmodel.wm.utils import load_pretrained, save_pretrained

    cfg = _offline_config()
    model = TDMPC2(cfg).to(dtype).eval()
    spec = OmegaConf.create(
        {
            '_target_': 'stable_worldmodel.wm.tdmpc2.TDMPC2',
            '_recursive_': False,
            'cfg': cfg,
        }
    )
    save_pretrained(model, 'offline', config=spec, cache_dir=tmp_path)
    restored = load_pretrained('offline', cache_dir=tmp_path).to(dtype).eval()
    assert restored.cfg.preprocessing == cfg.preprocessing

    # No stats dataset is needed: fitting on held-out data must not occur.
    policy = make_model_policy(_eval_config(), restored, stats_dataset=None)
    assert policy.solver.cost is restored
    assert set(policy.process) == {'observation', 'goal_observation'}
    reference = deepcopy(model)
    del reference.cfg.preprocessing
    reference_policy = WorldModelPolicy(
        solver=CEMSolver(
            reference,
            num_samples=8,
            topk=2,
            n_steps=2,
            var_scale=0.01,
            seed=17,
        ),
        config=PlanConfig(horizon=3, receding_horizon=1),
    )
    env = SimpleNamespace(
        num_envs=2,
        action_space=Box(-1, 1, (2, 2)),
        single_action_space=Box(-1, 1, (2,)),
    )
    policy.set_env(env)
    reference_policy.set_env(env)
    info, prepared = _planning_inputs(cfg, raw=True)
    if with_time_dim:
        info = {key: value.unsqueeze(1) for key, value in info.items()}
    torch.manual_seed(29)
    expected = reference_policy.get_action(
        {key: value.numpy() for key, value in prepared.items()}
    )
    torch.manual_seed(29)
    actual = policy.get_action(
        {key: value.numpy() for key, value in info.items()}
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert np.isfinite(actual).all()


def test_eval_rejects_missing_training_statistics():
    pytest.importorskip('stable_pretraining')
    from scripts.plan.eval_wm import make_model_policy

    model = TDMPC2(_make_config()).eval()
    with pytest.raises(ValueError, match='no saved training preprocessing'):
        make_model_policy(_eval_config(), model, stats_dataset=None)


@pytest.mark.parametrize('field', ['action_block', 'history_len'])
def test_eval_rejects_unsupported_tdmpc2_temporal_config(field):
    pytest.importorskip('stable_pretraining')
    from scripts.plan.eval_wm import make_model_policy

    cfg = _eval_config()
    cfg.plan_config[field] = 2
    with pytest.raises(ValueError, match=f'{field}=1'):
        make_model_policy(cfg, TDMPC2(_offline_config()), stats_dataset=None)


@pytest.mark.parametrize('width', [1, 3])
def test_saved_column_statistics_match_training_transform(width):
    pytest.importorskip('stable_pretraining')
    from scripts.train.tdmpc2 import get_column_normalizer

    values = np.arange(4 * width, dtype=np.float32).reshape(4, width)
    values[-1] = np.nan
    dataset = SimpleNamespace(get_col_data=lambda _key: values)
    transform, stats = get_column_normalizer(dataset, 'state', 'state')
    clean = torch.from_numpy(values[:-1])
    expected = (clean - clean.mean(0)) / (clean.std(0) + 1e-2)
    output = transform({'state': clean.clone()})['state']
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    restored = (clean - torch.tensor(stats['mean'])) / torch.tensor(
        stats['std']
    )
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_other_models_keep_evaluation_scalers():
    pytest.importorskip('stable_pretraining')
    from scripts.plan.eval_wm import make_model_policy
    from stable_worldmodel.planning import ShootingCostEvaluator

    values = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    dataset = SimpleNamespace(get_col_data=lambda _key: values)
    model = torch.nn.Linear(2, 2).eval()
    policy = make_model_policy(_eval_config(), model, dataset)
    assert isinstance(policy.solver.cost, ShootingCostEvaluator)
    np.testing.assert_allclose(
        policy.process['observation'].transform(values),
        [[-1.0, -1.0], [1.0, 1.0]],
    )
    assert policy.process['goal_observation'] is policy.process['observation']
    np.testing.assert_allclose(
        policy.process['action'].inverse_transform(np.zeros((1, 2))),
        [[3.0, 5.0]],
    )
