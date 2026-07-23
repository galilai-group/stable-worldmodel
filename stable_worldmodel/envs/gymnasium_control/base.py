import gymnasium as gym
import numpy as np

from stable_worldmodel import spaces as swm_spaces


class GymControlWrapper(gym.Wrapper):
    """Base wrapper for Gymnasium classic-control envs with variation_space.

    Subclasses must set:
      - ``variation_space`` (swm_spaces.Dict) in ``__init__``.
      - ``DEFAULT_VARIATIONS`` (tuple of dotted keys) as a class attribute.
      - ``_apply_physical_variations(active)``: write the sampled values onto
        ``self.env.unwrapped``.
    """

    DEFAULT_VARIATIONS: tuple = ()
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, env_id, render_mode=None, **kwargs):
        env = gym.make(env_id, render_mode=render_mode, **kwargs)
        super().__init__(env)
        self.env_name = env_id
        self.variation_space = None

    def reset(self, seed=None, options=None):
        options = options or {}

        assert self.variation_space is not None, (
            'variation_space must be set by subclass before reset'
        )
        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=self.DEFAULT_VARIATIONS,
        )

        sampled_keys = options.get('variation', self.DEFAULT_VARIATIONS)
        explicit_keys = list(options.get('variation_values', {}).keys())
        active = list(sampled_keys) + explicit_keys

        self._apply_physical_variations(active)

        obs, info = super().reset(seed=seed, options=options)
        info['env_name'] = self.env_name
        info['state'] = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info['env_name'] = self.env_name
        info['state'] = np.asarray(obs, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def _apply_physical_variations(self, active):
        raise NotImplementedError

    def _color_swaps(self):
        """Return a list of ``(src_rgb_tuple, target_value_array)`` pairs.

        ``src_rgb_tuple`` is a uint8 RGB triple matching pixels emitted by the
        underlying env's ``render()`` (e.g. ``(0, 0, 0)`` for CartPole's cart).
        ``target_value_array`` is a length-3 array in ``[0, 1]`` from the
        variation_space. Override per-env; default is no-op.
        """
        return []

    def render(self):
        img = self.env.render()
        if img is None or not isinstance(img, np.ndarray) or img.ndim != 3:
            return img
        swaps = self._color_swaps()
        if not swaps:
            return img
        img = img.copy()
        for src, tgt in swaps:
            src_arr = np.asarray(src, dtype=np.uint8)
            tgt_arr = np.clip(np.asarray(tgt, dtype=np.float32), 0.0, 1.0)
            tgt_u8 = (tgt_arr * 255.0 + 0.5).astype(np.uint8)
            if np.array_equal(src_arr, tgt_u8):
                continue
            mask = (img[..., :3] == src_arr).all(axis=-1)
            img[mask, :3] = tgt_u8
        return img
