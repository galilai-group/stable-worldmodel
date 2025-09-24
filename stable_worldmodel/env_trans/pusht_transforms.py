from typing import List
import gymnasium as gym
import pygame
import numpy as np
import requests
from io import BytesIO
import os
import random
# color transform functions
# shape transform functions


def remove_body_by_id(space, target_id):
    """Remove a body (and its shapes) from a pymunk.Space by its my_id."""
    for body in list(space.bodies):
        if getattr(body, "id", None) == target_id:
            for shape in list(body.shapes):
                space.remove(shape)
            space.remove(body)
            return True
    return False


def get_body_from_id(space, target_id):
    """Get a body (and its shapes) from a pymunk.Space by its my_id."""
    for body in list(space.bodies):
        if getattr(body, "id", None) == target_id:
            return body
    return None


class BaseDeform(gym.Wrapper):
    def __init__(
        self,
        env,
        every_k_steps=-1,
        apply_on_reset=False,
        apply_before_step=False,
        apply_at_init=True,
    ):
        super().__init__(env)
        self.every_k_steps = every_k_steps
        self.apply_on_reset = apply_on_reset
        self.apply_before_step = apply_before_step
        self.apply_at_init = apply_at_init
        self._step = 0

        self.init_applied = False  # Track if deformation has been applied at init

    def deform(self):
        raise NotImplementedError("Deform method not implemented.")

    @property
    def should_update(self):
        should_init_deform = not self.init_applied and self.apply_at_init
        return (
            self.every_k_steps > 0 and self._step % self.every_k_steps == 0
        ) or should_init_deform

    def reset(self, *args, **kwargs):
        self._step = 0
        reset_res = self.env.reset(*args, **kwargs)
        if self.apply_on_reset or self.should_update:
            print(f"{self._step}")
            self.deform()
        return reset_res

    def step(self, action):
        if self.apply_before_step and self.should_update:
            self.deform()

        step_res = self.env.step(action)
        self._step += 1

        if not self.apply_before_step and self.should_update:
            self.deform()
        return step_res


class BackgroundDeform(BaseDeform):
    def __init__(
        self,
        env,
        image: str = None,
        noise_fn: bool = False,
        noise_fixed: bool = False,
        **kwargs,
    ):
        super().__init__(env, **kwargs)

        if not pygame.display.get_init():
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.display.init()
            pygame.display.set_mode((1, 1))

        self.image = None
        self.noise_fn = noise_fn
        self.noise_fixed = noise_fixed

        # download image if url
        if image is not None and not os.path.exists(image):
            print(f"Image {image} not found locally, downloading...")
            response = requests.get(image)
            response.raise_for_status()
            bio = BytesIO(response.content)
            bio.seek(0)
            response = requests.get(image)
            image = BytesIO(response.content)
            self.image = pygame.image.load(bio, "bg.png")

        elif os.path.exists(image):
            self.image = pygame.image.load(image)

        # if noise is provided sample noise
        elif noise_fn is not None:
            self.image = self.ndarray_2_surface(noise_fn())

        else:
            raise ValueError(
                "Either an image URL/PATH or a noise function must be provided."
            )

        if self.apply_at_init:
            self.deform()

    def ndarray_2_surface(self, img):
        if not isinstance(img, (np.ndarray)):
            raise ValueError("Input must be a NumPy array.")

        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("NumPy background must be HxWx3 or HxWx4")

        # Convert dtype to uint8 (auto-scale floats in [0,1])
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0 + 0.5).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8, copy=False)

        # pygame.surfarray.make_surface expects (W, H, C)
        if arr.shape[2] == 3:
            surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
            return surf
        else:  # RGBA
            rgb = arr[..., :3]
            a = arr[..., 3]
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1)).convert_alpha()
            alpha = pygame.surfarray.pixels_alpha(surf)
            alpha[:, :] = a.swapaxes(0, 1)
            del alpha
            return surf

    def deform(self):
        if self.image is not None:
            # resample noise if necessary
            if self.noise_fn and not self.noise_fixed:
                self.image = self.ndarray_2_surface(self.noise_fn())

            self.env.unwrapped.set_background(self.image)


class ColorDeform(BaseDeform):
    def __init__(
        self,
        env,
        colors: List
        | str = None,  # list of names in pygame.color.THECOLORS, rbga tuples, or hex strings
        target: List | str = "agent",  # agent, block, goal, walls
        share_colors=False,  # share colors between targets
        **kwargs,  # every_k_steps, apply_on_reset, apply_before_step, ...
    ):
        super().__init__(env, **kwargs)
        self.target = target
        self.colors = colors
        self.share_colors = share_colors

    def _rand_color(self):
        """Return a pygame.Color from the provided palette or pygame's full set."""
        if self.colors:
            if isinstance(self.colors, str) or len(self.colors) == 1:
                choice = self.colors
            else:
                choice = random.choice(self.colors)
            try:
                return (
                    pygame.Color(choice)
                    if isinstance(choice, str)
                    else pygame.Color(*choice)
                )

            except (ValueError, TypeError):
                return pygame.Color(*choice)
        else:
            return pygame.Color(random.choice(list(pygame.color.THECOLORS.keys())))

    def _deform_color(self, target_str, shared_color=None):
        # Access underlying env objects
        space = self.env.unwrapped.space
        agent = self.env.unwrapped.agent
        block = self.env.unwrapped.block
        random_color = shared_color if self.share_colors else self._rand_color()
        target_body = None

        if target_str == "agent":
            target_body = get_body_from_id(space, agent.id)
        elif target_str == "block":
            target_body = get_body_from_id(space, block.id)
        # else get from id
        else:
            target_body = get_body_from_id(space, target_str)

        if target_str == "goal":
            self.env.unwrapped.goal_color = random_color

        elif target_body is not None:
            for shape in target_body.shapes:
                if hasattr(shape, "color"):
                    shape.color = random_color
                else:
                    # Keep the helpful debug print from your original code
                    print(f"Shape {shape} does not have a color attribute.")

    def deform(self):
        shared_color = self._rand_color() if self.share_colors else None

        # Access underlying env objects
        if isinstance(self.target, str):
            self._deform_color(self.target, shared_color)
        elif isinstance(self.target, list):
            for target_str in self.target:
                self._deform_color(target_str, shared_color)
        else:
            raise ValueError("Target must be a string or a list of strings.")


class ShapeDeform(BaseDeform):
    """
    Deform bodies into new shapes using env.unwrapped.add_shape.

    Parameters
    ----------
    target : str | list[str]
        "agent", "block", or an identifier handled by get_body_from_id.
        If a list, applies to each target in order.
    shapes : str | list[str]
        Single shape name (shared by all targets) or a list of shape names.
    randomize : bool
        If True and shapes is a list, sample a random shape for each target independently.
        If False and shapes is a list, pair by index (len(shapes) must match len(targets)).
    angle : float
        Rotation passed to add_shape (default 0).
    scale : float
        Scale passed to add_shape (default 10).

    Scheduling behavior is inherited from BaseDeform: apply_on_reset, every_k_steps, apply_before_step.
    """

    def __init__(
        self,
        env,
        shapes=None,
        target="block",
        randomize=False,
        angle: float = 0.0,
        scales: float | List[float] = 40.0,
        **kwargs,  # every_k_steps, apply_on_reset, apply_before_step, ...
    ):
        super().__init__(env, **kwargs)
        self.shapes = shapes or ["T", "I", "L", "Z", "square", "small_tee", "+"]
        self.target = target
        self.randomize = randomize
        self.angle = angle
        self.scales = scales

    # ---------- helpers ----------
    def _resolve_targets(self):
        if isinstance(self.target, str):
            return [self.target]
        if isinstance(self.target, (list, tuple)):
            return list(self.target)
        raise ValueError("target must be a str or a list[str].")

    def _choose_shape_for_target(self, idx: int) -> str:
        if isinstance(self.shapes, str):
            return self.shapes

        if isinstance(self.shapes, (list, tuple)):
            if self.randomize:
                return random.choice(self.shapes)
            if idx >= len(self.shapes):
                raise ValueError(
                    "When randomize=False and shapes is a list, "
                    "len(shapes) must match len(target)."
                )
            return self.shapes[idx]

        raise ValueError("shapes must be a str or a list[str].")

    def _lookup_body_for_target(self, t):
        space = self.env.unwrapped.space
        if t == "agent":
            return self.env.unwrapped.agent, "agent"
        elif t == "block":
            return self.env.unwrapped.block, "block"
        else:
            return get_body_from_id(space, t), t  # Assume t is an id

    def _body_color(self, body, default="Red"):
        try:
            shp = next(iter(body.shapes))
            return getattr(shp, "color", pygame.Color(default))
        except StopIteration:
            return pygame.Color(default)
        except Exception:
            return pygame.Color(default)

    # ---------- deform ----------
    def deform(self):
        space = self.env.unwrapped.space
        targets = self._resolve_targets()

        for i, t in enumerate(targets):
            body, known_label = self._lookup_body_for_target(t)
            if body is None:
                print(f"[ShapeDeform] Skipping target '{t}': no body found.")
                continue

            pos = getattr(body, "position", None)
            if pos is None:
                print(f"[ShapeDeform] Skipping target '{t}': body has no position.")
                continue

            color = self._body_color(body)
            new_shape_name = self._choose_shape_for_target(i)

            remove_body_by_id(space, getattr(body, "id", None))

            scale = self.scales if type(self.scales) in (int, float) else self.scales[i]

            # Create replacement body
            new_body = self.env.unwrapped.add_shape(
                new_shape_name, pos, self.angle, color=color, scale=scale
            )

            # Update env references when we know them
            if known_label == "agent":
                self.env.unwrapped.agent = new_body
            elif known_label == "block":
                self.env.unwrapped.block = new_body


# todo add TextureDeform
# todo add NoisyActionDeform
# todo add step_idx dependence scheduler
