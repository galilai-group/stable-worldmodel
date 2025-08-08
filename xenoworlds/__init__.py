import os

from gymnasium.envs.registration import register

from . import collect, data, evaluator, policy, solver, wrappers, predictor
from .utils import create_pil_image_from_url, set_state
from .wm import DummyWorldModel
from .world import World

register(
    id="xenoworlds/ImagePositioning-v1",
    entry_point="xenoworlds.envs.image_positioning:ImagePositioning",
)

register(
    id="xenoworlds/PushT-v1",
    entry_point="xenoworlds.envs.pusht:PushT",
)
