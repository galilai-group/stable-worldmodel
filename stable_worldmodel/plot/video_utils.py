from __future__ import annotations

from pathlib import Path

import numpy as np


def save_video(path: Path, frames: list[np.ndarray], fps: int = 15) -> None:
    if not frames:
        return
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    out = imageio.get_writer(str(path), fps=fps, codec='libx264')
    for f in frames:
        out.append_data(f)
    out.close()


def save_panel_videos(video_dir, panels, fps: int = 15) -> None:
    """Save one mp4 per env with labeled panels side-by-side.

    ``panels`` maps a label to per-env data indexable by env index. Each
    per-env entry is either a ``(T, H, W, C)`` sequence or a ``(H, W, C)``
    still that is repeated for every frame.
    """
    from PIL import Image, ImageDraw, ImageFont

    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    labels = list(panels)
    n_envs = len(panels[labels[0]])

    sample = np.asarray(panels[labels[0]][0])
    h, w = sample.shape[1:3] if sample.ndim == 4 else sample.shape[:2]
    n = len(labels)
    pad, gap, lh = max(12, w // 14), max(10, w // 16), max(22, w // 9)
    cw = (2 * pad + n * w + (n - 1) * gap + 15) // 16 * 16
    ch = (2 * pad + h + lh + 15) // 16 * 16
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', max(12, w // 14))
    except OSError:
        font = ImageFont.load_default()
    y_text = pad + h + max(8, lh // 4)

    for i in range(n_envs):
        env_panels = [np.asarray(panels[label][i]) for label in labels]
        T = max((len(p) for p in env_panels if p.ndim == 4), default=1)
        composed = []
        for t in range(T):
            c = np.full((ch, cw, 3), 250, dtype=np.uint8)
            for j, p in enumerate(env_panels):
                frame = p[min(t, len(p) - 1)] if p.ndim == 4 else p
                x = pad + j * (w + gap)
                c[pad : pad + h, x : x + w] = frame
            img = Image.fromarray(c)
            draw = ImageDraw.Draw(img)
            for j, label in enumerate(labels):
                b = draw.textbbox((0, 0), label, font=font)
                x = pad + j * (w + gap) + w // 2 - (b[2] - b[0]) // 2
                draw.text((x, y_text), label, fill=(130, 130, 130), font=font)
            composed.append(np.array(img))
        save_video(video_dir / f'env_{i}.mp4', composed, fps=fps)
