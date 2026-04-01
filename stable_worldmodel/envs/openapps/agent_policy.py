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
from .executor import VIEWPORT_HEIGHT, VIEWPORT_WIDTH, action_str_to_box5


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
        self._text_action_count = 0
        self._total_steps = 0

    def get_action(self, obs, **kwargs) -> np.ndarray:
        """Get action from VLM given the current observation.

        Args:
            obs: Dict with at least 'pixels' key (H, W, 3 uint8 array).

        Returns:
            Box(5,) float32 action vector.
        """
        screenshot = obs["pixels"] if isinstance(obs, dict) else obs
        self.history.append(screenshot)
        self._total_steps += 1

        # Call the VLM
        raw = self.agent.predict(
            screenshot=screenshot,
            task=self.task,
            history=list(self.history),
        )

        # Parse the VLM output into a BrowserGym action string
        parsed = self._parse_action(raw)

        # Handle text actions as no-ops (design notes §4.2)
        if parsed.startswith("keyboard_type") or parsed.startswith("type("):
            self._text_action_count += 1
            logger.debug(
                f"Text action dropped ({self._text_action_count}/"
                f"{self._total_steps}): {parsed}"
            )
            return np.zeros(5, dtype=np.float32)

        return action_str_to_box5(parsed)

    def _parse_action(self, raw_action: str) -> str:
        """Parse raw VLM output into a BrowserGym-style action string.

        Attempts to use uitars_parser if available, otherwise does basic
        string matching.
        """
        try:
            # Try uitars_parser from browsergym if available
            from browsergym.core.action.parsers import uitars_parser

            result = uitars_parser({"action": raw_action})
            return result["action"]
        except (ImportError, Exception):
            # Fallback: return raw string if it already looks like an action
            return raw_action

    @property
    def text_action_rate(self) -> float:
        """Fraction of steps where text actions were dropped."""
        if self._total_steps == 0:
            return 0.0
        return self._text_action_count / self._total_steps


class DummyPolicy(BasePolicy):
    """Random click policy for smoke-testing without a VLM."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "dummy"

    def get_action(self, obs, **kwargs) -> np.ndarray:
        """Return a random click action."""
        vec = np.zeros(5, dtype=np.float32)
        vec[0] = np.random.uniform(0.0, 0.33)  # always click
        vec[1] = np.random.uniform(0.0, 1.0)   # random x
        vec[2] = np.random.uniform(0.0, 1.0)   # random y
        return vec
