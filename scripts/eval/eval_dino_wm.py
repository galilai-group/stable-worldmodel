import hydra
import lightning as pl
import minari
import stable_ssl as ssl
import torch
import xenoworlds
from einops import rearrange, repeat
from loguru import logger as logging
from stable_ssl.data import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel

from xenoworlds.predictor import CausalPredictor


class Config:
    """Configuration for the training script"""

    # encoder
    img_size: int = 224
    num_hist: int = 3
    num_pred: int = 1
    frameskip: int = 5
    num_patches: int = 1
    action_emb_dim: int = 10
    proprio_emb_dim: int = 10
    normalize_action: True


class Embedder(torch.nn.Module):
    def __init__(
        self,
        num_frames=1,
        tubelet_size=1,
        in_chans=8,
        emb_dim=10,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.patch_embed = torch.nn.Conv1d(
            in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)  # (B, T, B) -> (B, D, T)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x


class DINOWM(torch.nn.Module):
    def __init__(self, encoder, predictor, action_encoder, proprio_encoder):
        super().__init__()
        self.backbone = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder

    def forward(self, obs, actions, proprio):
        """world model forward pass"""

        def encode(obs, actions, proprio):
            # -- encode actions
            actions = self.action_encoder(actions)  # (B,T,A) -> (B,T,A_emb)

            # -- encode proprioceptive
            proprio = self.proprio_encoder(proprio)  # (B,T,P) -> (B,T,P_emb)

            # -- encode observations to get states
            B, T, C, H, W = obs.shape
            obs = rearrange(obs, "b t ... -> (b t) ...")

            # get the state
            state = self.backbone(obs).last_hidden_state  # (B*T, n_patches, D)
            state = state[:, 1:, :]  # drop cls token
            state = z = rearrange(state, "(b t) p d -> b t p d", b=B)

            # -- merge state, action, proprio
            n_patches = state.shape[2]
            # share action/proprio embedding accros patches for each time step
            proprio_repeat = repeat(
                proprio.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches
            )
            actions_repeat = repeat(
                actions.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches
            )
            # z (B, T, P, dim+A_emb+P_emb)
            z = torch.cat([state, actions_repeat, proprio_repeat], dim=3)
            return z

        # -- predict the next state
        def predict(z):
            T = z.shape[1]
            z = rearrange(z, "b t p d -> b (t p) d")
            preds = self.predictor(z)
            preds = rearrange(preds, "b (t p) d -> b t p d", t=T)
            return preds

        # -- preprocess inputs
        if type(actions) is dict:
            actions = [a.flatten(2) for a in actions.values()]
            actions = torch.cat(actions, -1)
        else:
            actions.flatten(2)

        # -- compute prediction error
        z = encode(obs, actions, proprio)
        z_preds = predict(z)

        # TODO should check from their code
        # z_src = z[:, : Config.num_hist, :, :]
        # z_tgt = z[:, Config.num_pred :, :, :]

        return (z, z_preds)


def get_world_model(action_dim, proprio_dim):
    """Return stable_ssl module with world model"""

    config = AutoConfig.from_pretrained("facebook/dinov2-small")
    encoder = AutoModel.from_config(config)
    emb_dim = config.hidden_size
    num_patches = (Config.img_size // config.patch_size) ** 2

    logging.info(f"Encoder: {encoder}, emb_dim: {emb_dim}, num_patches: {num_patches}")

    encoder = encoder.require_grad_(False)
    encoder.eval()

    # -- create predictor
    predictor = CausalPredictor(
        num_patches=num_patches,
        num_frames=Config.num_hist,
        dim=emb_dim + Config.proprio_emb_dim + Config.action_emb_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        pool="mean",
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    logging.info(f"Predictor: {predictor}")

    # -- create action encoder
    action_encoder = Embedder(in_chans=action_dim, emb_dim=Config.action_emb_dim)
    logging.info(f"Action dim: {action_dim}, action emb dim: {Config.action_emb_dim}")

    # -- create proprioceptive encoder
    proprio_encoder = Embedder(in_chans=proprio_dim, emb_dim=Config.proprio_emb_dim)
    logging.info(
        f"Proprio dim: {proprio_dim}, proprio emb dim: {Config.proprio_emb_dim}"
    )

    # NOTE: can add a decoder here if needed

    # -- world model as a stable_ssl module
    world_model = ssl.Module(
        backbone=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        forward=forward,
    )

    return world_model


@hydra.main(version_base=None, config_path="./", config_name="slurm")
def run(cfg):
    """Run training of predictor"""

    def load_ckpt(module, name):
        module.load_state_dict(torch.load(name, map_location="cpu"))
        return

    # -- make transform operations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        transforms.ToImage(
            mean=mean,
            std=std,
            source="observations.pixels",
            target="observations.pixels",
        ),
    )

    wrappers = [
        # lambda x: RecordVideo(x, video_folder="./videos"),
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]

    world = xenoworlds.World(
        "xenoworlds/PushT-v1", num_envs=4, wrappers=wrappers, max_episode_steps=100
    )

    world_model = get_world_model(action_dim, proprio_dim)

    world_model = xenoworlds.DummyWorldModel(
        image_shape=(3, 224, 224), action_dim=world.single_action_space.shape[0]
    )

    # -- create a planning policy with a gradient descent solver
    # solver = xenoworlds.solver.GDSolver(world_model, n_steps=100, action_space=world.action_space)
    # policy = xenoworlds.policy.PlanningPolicy(world, solver)
    # -- create a random policy
    policy = xenoworlds.policy.RandomPolicy(world)

    # -- run evaluation
    evaluator = xenoworlds.evaluator.Evaluator(world, policy)
    data = evaluator.run(episodes=5)

    # data will be a dict with all the collected metrics

    # # visualize a rollout video (e.g. for debugging purposes)
    # xenoworlds.utils.save_rollout_videos(data["frames_list"])


if __name__ == "__main__":
    run()
