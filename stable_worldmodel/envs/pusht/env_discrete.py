from gymnasium import spaces

from ..quantizer import PolarQuantizer
from .env import PushT


DEFAULT_VARIATIONS = ("agent.start_position", "block.start_position", "block.angle")


class PushTDiscrete(PushT):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,
        "render_fps": 10,
    }
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        quantizer=None,
        block_cog=None,
        damping=None,
        render_action=False,
        resolution=224,
        with_target=True,
        render_mode="rgb_array",
        relative=True,
        init_value=None,
    ):
        super().__init__(
            block_cog=block_cog,
            damping=damping,
            render_action=render_action,
            resolution=resolution,
            with_target=with_target,
            render_mode=render_mode,
            relative=relative,
            init_value=init_value,
        )

        # override action space to be discrete
        # rem: new actions are discrete bin
        self.quantizer = quantizer or PolarQuantizer(16, 16, max_action_distance=1.0)
        self.action_space = spaces.MultiDiscrete(self.quantizer.action_shape)

    def step(self, quantized_action):
        action = self.quantizer.dequantize(quantized_action)
        return super().step(action)
