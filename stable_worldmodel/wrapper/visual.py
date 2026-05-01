import math
import pathlib

import cv2
import gymnasium as gym
import imageio.v3 as iio
import numpy as np


VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.webm', '.gif', '.mkv'}


def _load_media(path):
    path = pathlib.Path(path)
    if path.suffix.lower() in VIDEO_EXTS:
        return np.stack(list(iio.imiter(path)))
    return np.asarray(iio.imread(path))


def constant(value):
    """Schedule that always returns ``value``."""
    return lambda step: value


def linear(start, end, horizon):
    """Linear ramp from ``start`` to ``end`` over ``horizon`` steps; held at ``end`` after."""
    return lambda step: (
        start + (end - start) * min(step / max(1, horizon), 1.0)
    )


def cosine(start, end, horizon):
    """Cosine ramp from ``start`` to ``end`` over ``horizon`` steps; held at ``end`` after."""

    def f(step):
        t = min(step / max(1, horizon), 1.0)
        return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))

    return f


def exponential(start, decay, floor=0.0):
    """Exponential decay ``start * decay**step``, lower-bounded by ``floor``."""
    return lambda step: max(start * (decay**step), floor)


def sinusoidal(low, high, period):
    """Sinusoid oscillating in ``[low, high]`` with the given ``period`` (in steps)."""
    amp = 0.5 * (high - low)
    mid = 0.5 * (high + low)
    return lambda step: (
        mid + amp * math.sin(2 * math.pi * step / max(1.0, period))
    )


class _PixelTransform(gym.Wrapper):
    """Base class that applies ``_apply`` to ``render()`` output and to ``info['pixels*']``."""

    def _apply(self, frame):
        raise NotImplementedError

    def _apply_to_info(self, info):
        for k, v in info.items():
            if k == 'pixels' or k.startswith('pixels.'):
                info[k] = self._apply(v)
        return info

    def render(self):
        frame = self.env.render()
        return frame if frame is None else self._apply(frame)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._apply_to_info(info)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward, term, trunc, self._apply_to_info(info)


class ChromaKeyWrapper(gym.Wrapper):
    """Replace pixels matching a key color in rendered frames with an image or video background.

    Works like a green-screen: pixels close to ``key_color`` (within ``tolerance``) are swapped
    out for the corresponding pixels of ``media``. If ``media`` is a video, frames advance and
    loop on each call to ``render``.
    """

    def __init__(self, env, key_color, media, tolerance=0.0):
        super().__init__(env)
        self._keys = np.atleast_2d(
            np.asarray(key_color, dtype=np.int16)
        ).reshape(-1, 3)
        self._tol = float(tolerance)
        media = (
            _load_media(media)
            if isinstance(media, (str, pathlib.Path))
            else np.asarray(media)
        )
        self._is_video = media.ndim == 4
        self._media = media
        self._idx = 0

    def _next_frame(self, h, w):
        frame = self._media[self._idx] if self._is_video else self._media
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        if self._is_video:
            self._idx = (self._idx + 1) % len(self._media)
        return frame

    def _apply(self, frame):
        h, w = frame.shape[:2]
        diff = frame.astype(np.int16) - self._keys[:, None, None, :]
        mask = (np.linalg.norm(diff, axis=-1) <= self._tol).any(axis=0)
        out = frame.copy()
        out[mask] = self._next_frame(h, w)[mask]
        return out

    def _apply_to_info(self, info):
        for k, v in info.items():
            if k == 'pixels' or k.startswith('pixels.'):
                info[k] = self._apply(v)
        return info

    def render(self):
        frame = self.env.render()
        return frame if frame is None else self._apply(frame)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._apply_to_info(info)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward, term, trunc, self._apply_to_info(info)


class NoiseWrapper(_PixelTransform):
    """Add Gaussian pixel noise with a step-dependent standard deviation.

    ``std`` is either a float or a callable ``f(step) -> float`` (e.g. ``linear``,
    ``cosine``, ``exponential``, ``sinusoidal``, or any user-provided function).
    The wrapper increments an internal step counter on each ``env.step`` call and
    passes the current count to the schedule before sampling noise.
    """

    def __init__(self, env, std=10.0, seed=None):
        super().__init__(env)
        self._std = std if callable(std) else constant(std)
        self._rng = np.random.default_rng(seed)
        self._step = 0

    @property
    def step_count(self):
        return self._step

    def _apply(self, frame):
        s = float(self._std(self._step))
        if s <= 0:
            return frame
        noise = self._rng.normal(0, s, frame.shape)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(
            frame.dtype
        )

    def step(self, action):
        out = super().step(action)
        self._step += 1
        return out


class ColorJitterWrapper(_PixelTransform):
    """Random brightness, contrast, saturation, and hue shifts. Factors resample each reset."""

    def __init__(
        self,
        env,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        seed=None,
    ):
        super().__init__(env)
        self._b = float(brightness)
        self._c = float(contrast)
        self._s = float(saturation)
        self._h = float(hue)
        self._rng = np.random.default_rng(seed)
        self._sample()

    def _sample(self):
        self._db = self._rng.uniform(max(0.0, 1 - self._b), 1 + self._b)
        self._dc = self._rng.uniform(max(0.0, 1 - self._c), 1 + self._c)
        self._ds = self._rng.uniform(max(0.0, 1 - self._s), 1 + self._s)
        self._dh = self._rng.uniform(-self._h, self._h)

    def _apply(self, frame):
        f = frame.astype(np.float32) * self._db
        mean = f.mean(axis=(0, 1), keepdims=True)
        f = (f - mean) * self._dc + mean
        rgb = np.clip(f, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + self._dh * 180.0) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * self._ds, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return out.astype(frame.dtype)

    def reset(self, **kwargs):
        self._sample()
        return super().reset(**kwargs)


class BlurWrapper(_PixelTransform):
    """Gaussian blur with odd ``kernel`` size and standard deviation ``sigma`` (0 = derived from kernel)."""

    def __init__(self, env, kernel=5, sigma=0.0):
        super().__init__(env)
        self._kernel = int(kernel) | 1
        self._sigma = float(sigma)

    def _apply(self, frame):
        return cv2.GaussianBlur(
            frame, (self._kernel, self._kernel), self._sigma
        )


class OcclusionWrapper(_PixelTransform):
    """Cover the frame with ``num_patches`` rectangles of fractional ``size``. Patches resample per reset."""

    def __init__(
        self, env, num_patches=1, size=(0.1, 0.3), color=0, seed=None
    ):
        super().__init__(env)
        self._n = int(num_patches)
        self._size = (float(size[0]), float(size[1]))
        self._color = color
        self._rng = np.random.default_rng(seed)
        self._patches = None

    def _sample(self, h, w):
        out = []
        for _ in range(self._n):
            ph = max(1, int(self._rng.uniform(*self._size) * h))
            pw = max(1, int(self._rng.uniform(*self._size) * w))
            y = int(self._rng.integers(0, max(1, h - ph + 1)))
            x = int(self._rng.integers(0, max(1, w - pw + 1)))
            out.append((y, x, ph, pw))
        return out

    def _apply(self, frame):
        h, w = frame.shape[:2]
        if self._patches is None:
            self._patches = self._sample(h, w)
        out = frame.copy()
        for y, x, ph, pw in self._patches:
            out[y : y + ph, x : x + pw] = self._color
        return out

    def reset(self, **kwargs):
        self._patches = None
        return super().reset(**kwargs)


class MovingPatchWrapper(_PixelTransform):
    """Overlay ``num_patches`` solid-color rectangles that drift with their own velocity.

    Each patch has an independent position and velocity sampled at reset. Positions advance
    by their velocity once per ``env.step`` and reflect off the frame edges, so motion is
    smooth and continuous (no teleporting). ``speed`` is the velocity magnitude in pixels
    per step.
    """

    def __init__(
        self,
        env,
        num_patches=1,
        size=(0.1, 0.2),
        color=255,
        speed=2.0,
        seed=None,
    ):
        super().__init__(env)
        self._n = int(num_patches)
        self._size = (float(size[0]), float(size[1]))
        self._color = color
        self._speed = float(speed)
        self._rng = np.random.default_rng(seed)
        self._patches = None
        self._frame_shape = None

    def _init_patches(self, h, w):
        out = []
        for _ in range(self._n):
            ph = max(1, int(self._rng.uniform(*self._size) * h))
            pw = max(1, int(self._rng.uniform(*self._size) * w))
            y = float(self._rng.uniform(0, max(1, h - ph)))
            x = float(self._rng.uniform(0, max(1, w - pw)))
            angle = float(self._rng.uniform(0, 2 * math.pi))
            vy = self._speed * math.sin(angle)
            vx = self._speed * math.cos(angle)
            out.append([y, x, ph, pw, vy, vx])
        return out

    def _advance(self):
        if self._patches is None or self._frame_shape is None:
            return
        h, w = self._frame_shape
        for p in self._patches:
            p[0] += p[4]
            p[1] += p[5]
            ymax = h - p[2]
            xmax = w - p[3]
            if p[0] < 0:
                p[0] = -p[0]
                p[4] = -p[4]
            elif p[0] > ymax:
                p[0] = 2 * ymax - p[0]
                p[4] = -p[4]
            if p[1] < 0:
                p[1] = -p[1]
                p[5] = -p[5]
            elif p[1] > xmax:
                p[1] = 2 * xmax - p[1]
                p[5] = -p[5]

    def _apply(self, frame):
        h, w = frame.shape[:2]
        if self._patches is None:
            self._frame_shape = (h, w)
            self._patches = self._init_patches(h, w)
        out = frame.copy()
        for y, x, ph, pw, _, _ in self._patches:
            yi, xi = int(y), int(x)
            out[yi : yi + ph, xi : xi + pw] = self._color
        return out

    def reset(self, **kwargs):
        self._patches = None
        self._frame_shape = None
        return super().reset(**kwargs)

    def step(self, action):
        out = super().step(action)
        self._advance()
        return out


class RandomShiftWrapper(_PixelTransform):
    """DrQ-style random shift: replicate-pad by ``pad`` pixels then random crop back. Resamples each call."""

    def __init__(self, env, pad=4, seed=None):
        super().__init__(env)
        self._pad = int(pad)
        self._rng = np.random.default_rng(seed)

    def _apply(self, frame):
        p = self._pad
        padded = cv2.copyMakeBorder(frame, p, p, p, p, cv2.BORDER_REPLICATE)
        dy = int(self._rng.integers(0, 2 * p + 1))
        dx = int(self._rng.integers(0, 2 * p + 1))
        h, w = frame.shape[:2]
        return padded[dy : dy + h, dx : dx + w]


class CutoutWrapper(_PixelTransform):
    """Mask ``num`` random rectangles per frame with ``color``. Resamples on every call."""

    def __init__(self, env, num=1, size=(0.1, 0.2), color=0, seed=None):
        super().__init__(env)
        self._n = int(num)
        self._size = (float(size[0]), float(size[1]))
        self._color = color
        self._rng = np.random.default_rng(seed)

    def _apply(self, frame):
        h, w = frame.shape[:2]
        out = frame.copy()
        for _ in range(self._n):
            ch = max(1, int(self._rng.uniform(*self._size) * h))
            cw = max(1, int(self._rng.uniform(*self._size) * w))
            y = int(self._rng.integers(0, max(1, h - ch + 1)))
            x = int(self._rng.integers(0, max(1, w - cw + 1)))
            out[y : y + ch, x : x + cw] = self._color
        return out


class RandomConvWrapper(_PixelTransform):
    """Pass the frame through a randomly-initialized conv (3->3 channels). Weights resample per reset."""

    def __init__(self, env, kernel_size=3, seed=None):
        super().__init__(env)
        self._k = int(kernel_size) | 1
        self._rng = np.random.default_rng(seed)
        self._sample()

    def _sample(self):
        scale = 1.0 / (self._k * self._k * 3)
        self._weights = self._rng.normal(
            0, math.sqrt(scale), (3, self._k, self._k, 3)
        ).astype(np.float32)

    def _apply(self, frame):
        f = frame.astype(np.float32)
        out = np.zeros_like(f)
        for o in range(3):
            for i in range(3):
                out[..., o] += cv2.filter2D(
                    f[..., i], -1, self._weights[o, :, :, i]
                )
        return np.clip(out, 0, 255).astype(frame.dtype)

    def reset(self, **kwargs):
        self._sample()
        return super().reset(**kwargs)


class GrayscaleWrapper(_PixelTransform):
    """Convert frame to grayscale. ``keep_channels=True`` broadcasts back to 3 channels."""

    def __init__(self, env, keep_channels=True):
        super().__init__(env)
        self._keep = bool(keep_channels)

    def _apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self._keep:
            gray = np.stack([gray] * 3, axis=-1)
        return gray


class ResolutionWrapper(_PixelTransform):
    """Downsample the frame by ``scale`` then upsample back to the original size."""

    def __init__(self, env, scale=0.5):
        super().__init__(env)
        self._scale = float(scale)

    def _apply(self, frame):
        h, w = frame.shape[:2]
        sw = max(1, int(round(w * self._scale)))
        sh = max(1, int(round(h * self._scale)))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
