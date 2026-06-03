"""Policies for the OpenApps MCP env.

VLMPolicy emits raw UI-TARS / BrowserGym action *strings*; the env routes
them to the server's ``act_str`` (which parses both formats), so there is
no grid codec or action dropping here anymore. DummyPolicy samples the
env's pixel-native Dict action space.
"""

from collections import deque

import numpy as np
from loguru import logger

from stable_worldmodel.policy import BasePolicy

try:
    # OpenApps' own UI-TARS -> BrowserGym translation, so the mapping is
    # identical to running UI-TARS directly against OpenApps.
    from open_apps.agent.utils import uitars_parser
except Exception:  # pragma: no cover - optional heavy dep (agentlab)
    uitars_parser = None


class VLMPolicy(BasePolicy):
    """Policy that uses a Vision-Language Model to select UI actions.

    Args:
        agent: Object with ``predict(screenshot, task, history)`` returning a
            UI-TARS- or BrowserGym-formatted action string.
        task_description: Natural-language goal for the VLM prompt.
        history_len: Number of past screenshots to include in the prompt.
    """

    def __init__(self, agent, task_description: str, history_len: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.type = 'vlm'
        self.agent = agent
        self.task = task_description
        self.history: deque = deque(maxlen=history_len)
        self._total_steps = 0

    def get_action(self, obs, **kwargs) -> str:
        """Return a BrowserGym action string for the env's ``act`` path."""
        screenshot = obs['pixels'] if isinstance(obs, dict) else obs
        screenshot = np.squeeze(screenshot)
        self.history.append(screenshot)
        self._total_steps += 1

        raw = self.agent.predict(
            screenshot=screenshot,
            task=self.task,
            history=list(self.history),
        )
        action = uitars_parser({'action': raw})['action'] if uitars_parser else raw
        logger.debug(f'VLM action: {raw!r} -> {action!r}')
        return action


class DummyPolicy(BasePolicy):
    """Random policy: samples the env's Dict action space."""

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.type = 'dummy'
        self.seed = seed

    def get_action(self, obs, **kwargs):
        return self.env.action_space.sample()

    def set_seed(self, seed: int) -> None:
        if self.env is not None:
            self.env.action_space.seed(seed)
