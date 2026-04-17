"""VLMPolicy — BasePolicy subclass that calls VLMs for action selection.

Implements get_action(obs, **kwargs) -> np.ndarray, which is the only
interface swm calls. The VLM complexity (screenshot encoding, prompt
construction, action parsing) is fully hidden from swm.

Supported agents: UI-TARS-1.5-7B, GPT-4o/5.1, Claude, Dummy (random).
"""

from collections import deque

import numpy as np
from loguru import logger

from stable_worldmodel.policy import BasePolicy
from .executor import (
    GRID_X,
    GRID_Y,
    action_str_to_multidiscrete,
)


class VLMPolicy(BasePolicy):
    """Policy that uses a Vision-Language Model to select UI actions.

    Args:
        agent: Any object with a predict(screenshot, task, history) method
               that returns a TARS-formatted or BrowserGym-formatted action
               string.
        task_description: Natural-language description of the task goal.
        history_len: Number of past screenshots to include in the VLM prompt.
    """

    def __init__(
        self,
        agent,
        task_description: str,
        history_len: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = "vlm"
        self.agent = agent
        self.task = task_description
        self.history: deque = deque(maxlen=history_len)
        self._dropped_action_count = 0
        self._total_steps = 0

    def get_action(self, obs, **kwargs) -> np.ndarray:
        """Get action from VLM given the current observation.

        Args:
            obs: Dict with at least 'pixels' key (H, W, 3 uint8 array).

        Returns:
            MultiDiscrete int64 array [action_type, grid_x, grid_y].
        """
        screenshot = obs["pixels"] if isinstance(obs, dict) else obs
        # swm's vector-env wrapping adds a leading batch dim; openapps's
        # TarsAgent expects a plain HxWx3 frame, so normalize here.
        screenshot = np.squeeze(screenshot)
        self.history.append(screenshot)
        self._total_steps += 1

        raw = self.agent.predict(
            screenshot=screenshot,
            task=self.task,
            history=list(self.history),
        )

        parsed = self._parse_action(raw)

        # The env only supports click / scroll_up / scroll_down. Any other
        # TARS action (type, wait, finished, hotkey, drag, ...) is dropped
        # to a center no-op click so the episode can continue.
        if not self._is_supported(parsed):
            self._dropped_action_count += 1
            logger.debug(
                f"Unsupported action dropped ({self._dropped_action_count}/"
                f"{self._total_steps}): {parsed}"
            )
            return np.array([[0, GRID_X // 2, GRID_Y // 2]], dtype=np.int64)

        return action_str_to_multidiscrete(parsed)[None, :]

    @staticmethod
    def _is_supported(parsed: str) -> bool:
        """True if the parsed action maps to a click/scroll the env can run."""
        supported = ("mouse_click", "click(", "scroll")
        return parsed.startswith(supported)

    def _parse_action(self, raw_action: str) -> str:
        """Parse raw VLM output into a BrowserGym-style action string."""
        try:
            from browsergym.core.action.parsers import uitars_parser
            result = uitars_parser({"action": raw_action})
            return result["action"]
        except (ImportError, Exception):
            return raw_action

    @property
    def dropped_action_rate(self) -> float:
        """Fraction of steps where unsupported actions were dropped."""
        if self._total_steps == 0:
            return 0.0
        return self._dropped_action_count / self._total_steps


class DummyPolicy(BasePolicy):
    """Random action policy for smoke-testing without a VLM.

    Samples uniformly from MultiDiscrete([NUM_ACTIONS, GRID_X, GRID_Y]).
    """

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.type = "dummy"
        self.seed = seed

    def get_action(self, obs, **kwargs) -> np.ndarray:
        """Return a random MultiDiscrete action."""
        return self.env.action_space.sample()

    def set_seed(self, seed: int) -> None:
        if self.env is not None:
            self.env.action_space.seed(seed)
