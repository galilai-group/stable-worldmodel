"""
Game logic for Potion Lab environment.

Contains:
- Physics setup and configuration
- Collision handlers for player-tool interactions
- Rendering functions for essences with visual patterns
- Round configuration and management
"""

import json
import os
from typing import Any

import numpy as np
import pygame
import pymunk
from pymunk.vec2d import Vec2d

from .entities import (
    ESSENCE_TYPES,
    Bottler,
    Cauldron,
    CauldronState,
    DeliveryWindow,
    Dispenser,
    Enchanter,
    Essence,
    PhysicsConfig,
    Player,
    Refiner,
    ToolState,
    TrashCan,
)


# Rendering constants
PROGRESS_BAR_HEIGHT_SCALE = 0.2  # Progress bar height relative to tile_size
PROGRESS_BAR_WIDTH_SCALE = 0.8  # Progress bar width relative to tool width
PROGRESS_BAR_OFFSET_SCALE = 0.5  # Progress bar offset relative to tile_size
CAULDRON_SLOT_OFFSET_SCALE = 0.6  # Distance of cauldron slots from center
ESSENCE_RADIUS_SCALE_DEFAULT = 0.35  # Default radius for essence rendering


def _get_asset_path(asset_name: str) -> str:
    """Get absolute path to an asset."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Map tool class names to asset filenames
    asset_map = {
        "Enchanter": "enchanter.png",
        "Refiner": "refiner.png",
        "Cauldron": "cauldron.png",
        "Bottler": "bottler.png",
        "DeliveryWindow": "delivery_window.png",
        "Dispenser": "essence_dispenser.png",
        "Player": "wizard.png",
    }

    filename = asset_map.get(asset_name)
    if not filename:
        return None

    return os.path.join(current_dir, "assets", filename)


def _ensure_asset_loaded(obj, asset_name: str, size: tuple[float, float]) -> pygame.Surface | None:
    """
    Ensure the object has its asset loaded and cached.

    Args:
        obj: The game object (Tool, Dispenser, Player)
        asset_name: Name of the asset to load
        size: Target size (width, height)

    Returns:
        The cached pygame Surface or None if loading failed.
    """
    # Check if we already have the cached image and it's the right size
    cached_image = getattr(obj, "cached_image", None)
    cached_size = getattr(obj, "cached_size", None)

    # If cached image exists and size matches (tolerance for float precision), return it
    if cached_image and cached_size and abs(cached_size[0] - size[0]) < 1.0 and abs(cached_size[1] - size[1]) < 1.0:
        return cached_image

    # Load and scale
    path = _get_asset_path(asset_name)
    if not path:
        return None

    try:
        if not os.path.exists(path):
            return None

        image = pygame.image.load(path).convert_alpha()

        # Scale to fit while maintaining aspect ratio
        img_width, img_height = image.get_size()
        target_width, target_height = size

        scale_x = target_width / img_width
        scale_y = target_height / img_height
        scale = min(scale_x, scale_y)

        scaled_width = int(img_width * scale)
        scaled_height = int(img_height * scale)

        scaled_image = pygame.transform.smoothscale(image, (scaled_width, scaled_height))

        # Cache it on the object
        obj.cached_image = scaled_image
        obj.cached_size = size

        # Also clear any cached overlays/tints since base image changed
        if hasattr(obj, "cached_overlays"):
            obj.cached_overlays = {}

        return scaled_image
    except (pygame.error, FileNotFoundError):
        return None


def draw_tool(
    canvas: pygame.Surface,
    tool,
    custom_width: float | None = None,
    custom_height: float | None = None,
):
    """Draw a tool with its current state visualization using PNG assets."""
    pos = tool.position
    size = tool.size

    # Use custom dimensions if provided, otherwise use tool's size
    draw_width = custom_width if custom_width is not None else size[0]
    draw_height = custom_height if custom_height is not None else size[1]

    # Get the base image (cached)
    scaled_image = _ensure_asset_loaded(tool, tool.get_display_name(), (draw_width, draw_height))

    if scaled_image is not None:
        # Position centered
        img_rect = scaled_image.get_rect(center=(int(pos[0]), int(pos[1])))

        # Draw base image
        canvas.blit(scaled_image, img_rect)

        # Apply state overlays
        # We can cache overlays too if needed, but they are simple fills
        if hasattr(tool, "state"):
            overlay_color = None
            is_cauldron = isinstance(tool, Cauldron)

            if (is_cauldron and tool.state == CauldronState.STIRRING) or (
                not is_cauldron and tool.state == ToolState.PROCESSING
            ):
                overlay_color = (40, 40, 40, 100)  # Dark for processing
            elif tool.state == ToolState.DONE or tool.state == CauldronState.DONE:
                overlay_color = (0, 155, 0, 80)  # Green for done
            elif tool.state == CauldronState.READY_TO_STIR:
                overlay_color = (20, 20, 20, 60)  # Light shadow for ready

            if overlay_color:
                # Check for cached overlay
                overlay_key = f"state_{tool.state.name}"
                if not hasattr(tool, "cached_overlays"):
                    tool.cached_overlays = {}

                overlay = tool.cached_overlays.get(overlay_key)
                if not overlay:
                    overlay = pygame.Surface(scaled_image.get_size(), pygame.SRCALPHA)
                    overlay.fill(overlay_color)
                    tool.cached_overlays[overlay_key] = overlay

                canvas.blit(overlay, img_rect.topleft)

    else:
        # Fallback drawing
        rect = pygame.Rect(
            int(pos[0] - draw_width / 2), int(pos[1] - draw_height / 2), int(draw_width), int(draw_height)
        )
        pygame.draw.rect(canvas, tool.color, rect)
        pygame.draw.rect(canvas, (50, 50, 50), rect, 3)

    # Draw progress bar
    progress = _get_processing_progress(tool)
    if progress is not None:
        _draw_processing_bar(canvas, tool, progress)

    # Special rendering for Cauldron
    if isinstance(tool, Cauldron):
        _draw_cauldron_contents(canvas, tool)


def _draw_processing_bar(canvas: pygame.Surface, tool, progress: float):
    """Draw a small progress bar beneath a processing tool."""
    tile_size = tool.tile_size
    bar_width = int(tool.size[0] * PROGRESS_BAR_WIDTH_SCALE)
    bar_height = max(4, int(tile_size * PROGRESS_BAR_HEIGHT_SCALE))

    bar_x = int(tool.position[0] - bar_width / 2)
    bar_y = int(tool.position[1] + tool.size[1] / 2 - (tile_size * PROGRESS_BAR_OFFSET_SCALE))

    pygame.draw.rect(canvas, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
    progress_width = int(bar_width * progress)
    pygame.draw.rect(canvas, (100, 255, 100), (bar_x, bar_y, progress_width, bar_height))
    pygame.draw.rect(canvas, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), 1)


def _draw_cauldron_contents(canvas: pygame.Surface, cauldron):
    """Draw essences in the cauldron's 4 slots and stir progress."""
    pos = cauldron.position
    tile_size = cauldron.tile_size

    slot_offset = tile_size * CAULDRON_SLOT_OFFSET_SCALE
    essence_radius = int(tile_size * ESSENCE_RADIUS_SCALE_DEFAULT)

    # Positions for 4 slots: top, right, bottom, left
    slot_positions = [
        (pos[0], pos[1] - slot_offset),  # Top
        (pos[0] + slot_offset, pos[1]),  # Right
        (pos[0], pos[1] + slot_offset),  # Bottom
        (pos[0] - slot_offset, pos[1]),  # Left
    ]

    for i, essence_state in enumerate(cauldron.essence_slots):
        if essence_state is not None:
            render_essence(
                canvas,
                slot_positions[i],
                essence_radius,
                essence_state.essence_types,
                essence_state.enchanted_per_essence,
                essence_state.refined_per_essence,
                is_bottled=False,
            )

    # Draw stir progress bar
    if cauldron.state == CauldronState.STIRRING and cauldron.stir_time > 0:
        progress = cauldron.stir_progress / cauldron.stir_time
        bar_width = int(cauldron.size[0] * PROGRESS_BAR_WIDTH_SCALE)
        bar_height = max(4, int(tile_size * PROGRESS_BAR_HEIGHT_SCALE))

        bar_x = int(pos[0] - bar_width / 2)
        bar_y = int(pos[1] + cauldron.size[1] / 2 - (tile_size * PROGRESS_BAR_OFFSET_SCALE))

        pygame.draw.rect(canvas, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
        progress_width = int(bar_width * progress)
        pygame.draw.rect(canvas, (100, 255, 100), (bar_x, bar_y, progress_width, bar_height))
        pygame.draw.rect(canvas, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), 1)


def draw_player(canvas: pygame.Surface, player: Player):
    """Draw the player character using wizard asset."""
    pos = player.body.position
    render_size = getattr(player, "render_size", getattr(player, "size", 20))

    scaled_image = _ensure_asset_loaded(player, "Player", (render_size, render_size))

    if scaled_image is not None:
        img_rect = scaled_image.get_rect(center=(int(pos.x), int(pos.y)))
        canvas.blit(scaled_image, img_rect)
    else:
        # Fallback
        half_size = render_size / 2
        vertices = [
            (int(pos.x - half_size), int(pos.y - half_size)),
            (int(pos.x + half_size), int(pos.y - half_size)),
            (int(pos.x + half_size), int(pos.y + half_size)),
            (int(pos.x - half_size), int(pos.y + half_size)),
        ]
        pygame.draw.polygon(canvas, player.color, vertices)
        pygame.draw.polygon(canvas, (30, 30, 30), vertices, 3)


def _apply_essence_tint(image: pygame.Surface, tint_color: tuple[int, int, int]) -> pygame.Surface:
    """Apply essence color tinting to non-transparent pixels of an image."""
    # Create a copy to preserve alpha channel
    tinted_image = image.copy()

    # Create a color overlay surface
    tint_alpha = 0.7
    overlay = pygame.Surface(image.get_size(), pygame.SRCALPHA)
    tint_color_with_alpha = (*tint_color, int(255 * tint_alpha))
    overlay.fill(tint_color_with_alpha)

    # Blend the tint with the original image
    tinted_image.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    return tinted_image


def draw_dispenser(canvas: pygame.Surface, dispenser: Dispenser):
    """Draw a dispenser using PNG asset with essence color tinting."""
    pos = dispenser.position
    size = dispenser.size

    # Check for cached tinted image
    if not hasattr(dispenser, "cached_tinted_image"):
        # Load base asset first to ensure scaling
        _ensure_asset_loaded(dispenser, "Dispenser", size)

        # If we have base image, tint it and cache it
        if dispenser.cached_image:
            # This is a one-time tint operation per dispenser
            dispenser.cached_tinted_image = _apply_essence_tint(dispenser.cached_image, dispenser.color)
        else:
            dispenser.cached_tinted_image = None

    tinted_image = getattr(dispenser, "cached_tinted_image", None)

    if tinted_image is not None:
        img_rect = tinted_image.get_rect(center=(int(pos[0]), int(pos[1])))
        canvas.blit(tinted_image, img_rect)

        # Draw cooldown indicator
        if dispenser.cooldown > 0:
            alpha = int(255 * (dispenser.cooldown / dispenser.cooldown_duration))
            # Cache overlay surface if size/alpha predictable?
            # Alpha changes every frame, so simple surface creation is okay, or just reuse one surface and set alpha
            overlay = pygame.Surface(tinted_image.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, alpha // 2))
            canvas.blit(overlay, img_rect.topleft)
    else:
        # Fallback
        rect = pygame.Rect(int(pos[0] - size[0] / 2), int(pos[1] - size[1] / 2), int(size[0]), int(size[1]))
        pygame.draw.rect(canvas, dispenser.color, rect)
        pygame.draw.rect(canvas, (30, 30, 30), rect, 3)


def setup_physics_space() -> pymunk.Space:
    """Create and configure a pymunk physics space."""
    space = pymunk.Space()
    space.gravity = PhysicsConfig.GRAVITY
    space.damping = PhysicsConfig.DAMPING
    space.iterations = PhysicsConfig.ITERATIONS
    return space


def add_walls(space: pymunk.Space, map_width: float, map_height: float, wall_thickness: float = 10.0):
    """
    Add boundary walls to the physics space.

    Args:
        space: Pymunk space
        map_width: Width of the map in pixels
        map_height: Height of the map in pixels
        wall_thickness: Thickness of walls
    """
    walls = [
        # Bottom
        pymunk.Segment(space.static_body, (0, 0), (map_width, 0), wall_thickness),
        # Right
        pymunk.Segment(space.static_body, (map_width, 0), (map_width, map_height), wall_thickness),
        # Top
        pymunk.Segment(space.static_body, (map_width, map_height), (0, map_height), wall_thickness),
        # Left
        pymunk.Segment(space.static_body, (0, map_height), (0, 0), wall_thickness),
    ]

    for wall in walls:
        wall.friction = 1.0
        wall.elasticity = 0.0
        wall.collision_type = PhysicsConfig.LAYER_WALL

    space.add(*walls)
    return walls


# ============================================================================
# Collision Handlers
# ============================================================================


class CollisionHandler:
    """Manages collision detection and response between game objects using pymunk's collision handler system."""

    def __init__(self, env):
        """
        Initialize collision handler.

        Args:
            env: Reference to the PotionLabEnv
        """
        self.env = env
        self.player_stirring_cauldron = False
        self.essence_tool_collisions = set()

    def setup_handlers(self, space: pymunk.Space):
        """
        Set up pymunk collision handlers for game interactions.

        Args:
            space: The pymunk physics space
        """
        # Player-Dispenser collisions
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_PLAYER,
            collision_type_b=PhysicsConfig.LAYER_DISPENSER,
            begin=self._on_player_dispenser_begin,
        )

        # Player-Cauldron collisions
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_PLAYER,
            collision_type_b=PhysicsConfig.LAYER_CAULDRON,
            begin=self._on_player_cauldron_begin,
            separate=self._on_player_cauldron_separate,
        )

        # Essence-Tool collisions
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_ESSENCE,
            collision_type_b=PhysicsConfig.LAYER_TOOL,
            begin=self._on_essence_tool_begin,
        )

        # Essence-Cauldron collisions
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_ESSENCE,
            collision_type_b=PhysicsConfig.LAYER_CAULDRON,
            begin=self._on_essence_cauldron_begin,
        )

    def update(self):
        """Update collision state each frame."""
        # Handle ongoing cauldron stirring
        if self.player_stirring_cauldron:
            cauldron = self.env.tools.get("cauldron")
            if cauldron:
                cauldron.stir()

    def _on_player_dispenser_begin(self, arbiter, space, data):
        """Called when player collides with a dispenser."""
        # Get the dispenser from the collision
        for shape in arbiter.shapes:
            if hasattr(shape, "dispenser_obj"):
                dispenser = shape.dispenser_obj
                # Only allow interaction if dispenser is unlocked
                if dispenser.essence_type not in self.env.unlocked_dispensers:
                    return False  # Prevent collision

                essence_state = dispenser.dispense()

                if essence_state is not None:
                    # Check dispense quota and award reward
                    essence_type = dispenser.essence_type
                    if self.env.quotas["dispense"][essence_type] > 0:
                        self.env.pending_reward += self.env.reward_config["substep_reward"]
                        self.env.quotas["dispense"][essence_type] -= 1

                    # Spawn essence at dispenser location
                    position = dispenser.position
                    # Offset slightly so player can push it away
                    offset = Vec2d(0, dispenser.tile_size * self.env.dispenser_params["spawn_offset_multiplier"])
                    spawn_pos = (position[0] + offset.x, position[1] + offset.y)

                    # Get essence physics properties from variation space
                    mass = self.env.variation_space["essence"]["mass"].value
                    friction = self.env.variation_space["essence"]["friction"].value
                    elasticity = self.env.variation_space["essence"]["elasticity"].value
                    radius_scale = self.env.essence_config["radius_scale"]
                    drag_coefficient = self.env.essence_config["drag_coefficient"]

                    essence = Essence(
                        self.env.space,
                        spawn_pos,
                        essence_state,
                        self.env.tile_size,
                        mass,
                        friction,
                        elasticity,
                        radius_scale=radius_scale,
                        drag_coefficient=drag_coefficient,
                    )
                    self.env.essences.append(essence)
                break
        return True

    def _on_player_cauldron_begin(self, arbiter, space, data):
        """Called when player starts colliding with cauldron."""
        # Only allow interaction if cauldron is unlocked
        if "cauldron" not in self.env.unlocked_tools:
            return False  # Prevent collision

        cauldron = self.env.tools.get("cauldron")
        if cauldron:
            cauldron.start_stirring()
            self.player_stirring_cauldron = True

    def _on_player_cauldron_separate(self, arbiter, space, data):
        """Called when player stops colliding with cauldron."""
        # Only allow interaction if cauldron is unlocked
        if "cauldron" not in self.env.unlocked_tools:
            return False  # Prevent collision

        cauldron = self.env.tools.get("cauldron")
        if cauldron:
            cauldron.stop_stirring()
            self.player_stirring_cauldron = False

    def _on_essence_tool_begin(self, arbiter, space, data):
        """Called when essence collides with a tool."""
        # Find essence and tool from collision
        essence_obj = None
        tool_obj = None
        tool_name = None

        for shape in arbiter.shapes:
            if hasattr(shape, "essence_obj"):
                essence_obj = shape.essence_obj
            elif hasattr(shape, "tool_obj"):
                tool_obj = shape.tool_obj
                # Find the tool name
                for name, tool in self.env.tools.items():
                    if tool is tool_obj:
                        tool_name = name
                        break

        if essence_obj is None or tool_obj is None:
            return

        if tool_name != "delivery_window" and tool_name not in self.env.unlocked_tools:
            return False  # Prevent collision

        collision_key = (id(essence_obj), id(tool_obj))
        if collision_key in self.essence_tool_collisions:
            return

        if tool_name == "delivery_window":
            if hasattr(tool_obj, "validate_delivery"):
                # Check if this will be a valid delivery (matches a requirement)
                will_match = self._check_delivery_will_match(essence_obj, tool_obj)

                accepted = tool_obj.validate_delivery(essence_obj)
                if accepted:
                    self.essence_tool_collisions.add(collision_key)
                    if essence_obj in self.env.essences:
                        self.env.essences.remove(essence_obj)

                    # Award delivery reward only for valid matches
                    if will_match:
                        self.env.pending_reward += self.env.reward_config["substep_reward"]
        elif hasattr(tool_obj, "accept_essence"):
            accepted = tool_obj.accept_essence(essence_obj)
            if accepted:
                self.essence_tool_collisions.add(collision_key)
                if essence_obj in self.env.essences:
                    self.env.essences.remove(essence_obj)

    def _check_delivery_will_match(self, essence_obj, delivery_window) -> bool:
        """Check if an essence will match any unfulfilled requirement."""
        for requirement in delivery_window.required_items:
            if requirement.get("completed", False):
                continue
            if essence_obj.matches_requirement(requirement):
                return True
        return False

    def _on_essence_cauldron_begin(self, arbiter, space, data):
        """Called when essence collides with the cauldron."""
        # Only allow interaction if cauldron is unlocked
        if "cauldron" not in self.env.unlocked_tools:
            return False  # Prevent collision

        essence_obj = None

        for shape in arbiter.shapes:
            if hasattr(shape, "essence_obj"):
                essence_obj = shape.essence_obj
                break

        if essence_obj is None:
            return

        # Get cauldron
        cauldron = self.env.tools.get("cauldron")
        if cauldron is None:
            return

        collision_key = (id(essence_obj), id(cauldron))
        if collision_key in self.essence_tool_collisions:
            return

        if hasattr(cauldron, "accept_essence"):
            accepted = cauldron.accept_essence(essence_obj)
            if accepted:
                self.essence_tool_collisions.add(collision_key)
                if essence_obj in self.env.essences:
                    self.env.essences.remove(essence_obj)


# ============================================================================
# Rendering Functions
# ============================================================================


def render_essence(
    canvas: pygame.Surface,
    center: tuple[int, int],
    radius: float,
    essence_types: list[int],
    enchanted: list[bool],
    refined: list[bool],
    is_bottled: bool = False,
):
    """
    Unified function to render an essence (or combined essence) at a specific location.

    Args:
        canvas: Pygame surface to draw on
        center: (x, y) center position
        radius: Radius of the essence
        essence_types: List of essence type IDs
        enchanted: List of booleans indicating if each essence part is enchanted
        refined: List of booleans indicating if each essence part is refined
        is_bottled: Whether the essence is in a bottle
    """
    screen_pos = (int(center[0]), int(center[1]))
    screen_radius = int(radius)

    if is_bottled:
        bottle_rect = pygame.Rect(
            screen_pos[0] - screen_radius * 1.2,
            screen_pos[1] - screen_radius * 1.5,
            screen_radius * 2.4,
            screen_radius * 3,
        )
        pygame.draw.rect(canvas, (200, 200, 200), bottle_rect, 2)

        neck_rect = pygame.Rect(
            screen_pos[0] - screen_radius * 0.4,
            screen_pos[1] - screen_radius * 1.8,
            screen_radius * 0.8,
            screen_radius * 0.5,
        )
        pygame.draw.rect(canvas, (200, 200, 200), neck_rect, 2)

    if len(essence_types) > 1:
        n_parts = len(essence_types)
        angle_per_part = 360 / n_parts

        for i, essence_type in enumerate(essence_types):
            color = ESSENCE_TYPES[essence_type][1]

            # Draw pie slice
            start_angle = i * angle_per_part
            end_angle = (i + 1) * angle_per_part

            # Draw filled pie slice
            points = [screen_pos]
            for angle in range(int(start_angle), int(end_angle) + 1, 5):
                rad = np.deg2rad(angle)
                x = screen_pos[0] + radius * np.cos(rad)
                y = screen_pos[1] + radius * np.sin(rad)
                points.append((int(x), int(y)))
            points.append(screen_pos)

            if len(points) > 2:
                pygame.draw.polygon(canvas, color, points)

            # Draw patterns for this slice if needed
            if enchanted[i]:
                _draw_stripes_in_slice(canvas, screen_pos, radius, start_angle, end_angle)

            if refined[i]:
                _draw_dots_in_slice(canvas, screen_pos, radius, start_angle, end_angle)

    elif len(essence_types) == 1:
        color = ESSENCE_TYPES[essence_types[0]][1]
        pygame.draw.circle(canvas, color, screen_pos, screen_radius)

        # Draw patterns
        if enchanted[0]:
            _draw_stripes(canvas, screen_pos, screen_radius)

        if refined[0]:
            _draw_dots(canvas, screen_pos, screen_radius)


def draw_essence(canvas: pygame.Surface, essence: Essence):
    """
    Draw an essence with its visual patterns using the unified render function.
    Wrapper to take in an Essence object, whereas render_essence is also used for
    window, cauldron, etc.
    """
    render_essence(
        canvas,
        (int(essence.body.position.x), int(essence.body.position.y)),
        essence.radius,
        essence.state.essence_types,
        essence.state.enchanted_per_essence,
        essence.state.refined_per_essence,
        essence.state.is_bottled,
    )


def _draw_stripes(canvas: pygame.Surface, pos: tuple[int, int], radius: int):
    """Draw diagonal stripe pattern on a circle (enchanted)."""
    stripe_color = (0, 0, 0)

    # Draw diagonal lines clipped to circle
    spacing = max(3, radius // 5)
    line_width = max(1, radius // 15)

    for offset in range(-radius * 2, radius * 2, spacing):
        for y in range(-radius, radius + 1):
            x = y + offset
            # Check if point is within circle
            if x >= -radius and x <= radius:
                dist_sq = x * x + y * y
                if dist_sq <= radius * radius:
                    px = pos[0] + x
                    py = pos[1] + y
                    if abs(x - y - offset) < line_width:
                        canvas.set_at((int(px), int(py)), stripe_color)


def _draw_dots(canvas: pygame.Surface, pos: tuple[int, int], radius: int):
    """Draw dotted pattern on a circle (refined)."""
    dot_color = (0, 0, 0)

    # Draw dots in a grid pattern, only within circle bounds
    dot_radius = max(1, radius // 12)
    spacing = max(3, radius // 4)

    for x_offset in range(-radius + spacing // 2, radius, spacing):
        for y_offset in range(-radius + spacing // 2, radius, spacing):
            # Only draw if within circle
            dist = np.sqrt(x_offset**2 + y_offset**2)
            if dist <= radius - dot_radius - 2:
                dot_x = int(pos[0] + x_offset)
                dot_y = int(pos[1] + y_offset)
                pygame.draw.circle(canvas, dot_color, (dot_x, dot_y), dot_radius)


def _draw_stripes_in_slice(
    canvas: pygame.Surface,
    center: tuple[int, int],
    radius: float,
    start_angle: float,
    end_angle: float,
):
    """Draw diagonal stripes within a pie slice - looks like spliced top/bottom."""
    stripe_color = (0, 0, 0)

    # Draw diagonal stripes clipped to the pie slice
    # The stripes are diagonal (same as full circle), just clipped to the slice
    spacing = max(3, int(radius) // 5)
    line_width = max(1, int(radius) // 15)

    # Create list of points that define the pie slice
    slice_points = [(int(center[0]), int(center[1]))]
    for angle in range(int(start_angle), int(end_angle) + 1, 2):
        rad = np.deg2rad(angle)
        x = center[0] + radius * np.cos(rad)
        y = center[1] + radius * np.sin(rad)
        slice_points.append((int(x), int(y)))
    slice_points.append((int(center[0]), int(center[1])))

    # Draw diagonal stripes only within the pie slice
    for offset in range(-int(radius) * 2, int(radius) * 2, spacing):
        for y in range(-int(radius), int(radius) + 1):
            x = y + offset
            # Check if point is within circle
            if x >= -radius and x <= radius:
                dist_sq = x * x + y * y
                if dist_sq <= radius * radius:
                    px = int(center[0] + x)
                    py = int(center[1] + y)

                    # Check if point is within the pie slice using angle
                    angle_to_point = np.rad2deg(np.arctan2(y, x)) % 360
                    if start_angle <= angle_to_point <= end_angle or (
                        end_angle < start_angle and (angle_to_point >= start_angle or angle_to_point <= end_angle)
                    ):
                        if abs(x - y - offset) < line_width:
                            canvas.set_at((px, py), stripe_color)


def _draw_dots_in_slice(
    canvas: pygame.Surface,
    center: tuple[int, int],
    radius: float,
    start_angle: float,
    end_angle: float,
):
    """Draw dots within a pie slice - looks like spliced top/bottom."""
    # Use black for dots
    dot_color = (0, 0, 0)
    dot_radius = max(1, int(radius) // 12)
    spacing = max(3, int(radius) // 4)

    # Draw dots in grid pattern, only within the pie slice
    for x_offset in range(-int(radius) + spacing // 2, int(radius), spacing):
        for y_offset in range(-int(radius) + spacing // 2, int(radius), spacing):
            # Check if within circle
            dist = np.sqrt(x_offset**2 + y_offset**2)
            if dist <= radius - dot_radius - 2:
                # Check if point is within the pie slice using angle
                angle_to_point = np.rad2deg(np.arctan2(y_offset, x_offset)) % 360
                if start_angle <= angle_to_point <= end_angle or (
                    end_angle < start_angle and (angle_to_point >= start_angle or angle_to_point <= end_angle)
                ):
                    dot_x = int(center[0] + x_offset)
                    dot_y = int(center[1] + y_offset)
                    pygame.draw.circle(canvas, dot_color, (dot_x, dot_y), dot_radius)


def _get_processing_progress(tool) -> float | None:
    """
    Compute normalized processing progress for single-input tools.

    Returns:
        Progress in [0, 1] when the tool is actively processing, otherwise None.
    """
    if not isinstance(tool, Enchanter | Refiner | Bottler):
        return None

    if getattr(tool, "state", None) != ToolState.PROCESSING:
        return None

    total_time = None
    if isinstance(tool, Bottler):
        total_time = getattr(tool, "bottling_time", None)
    else:
        total_time = getattr(tool, "processing_time", None)

    if not total_time or total_time <= 0:
        return None

    remaining = max(0, getattr(tool, "timer", 0))
    progress = 1.0 - min(remaining, total_time) / total_time
    return max(0.0, min(1.0, progress))


def draw_ui(
    canvas: pygame.Surface,
    time_remaining: int,
    time_limit: int,
    requirements: list[dict],
    map_width: int,
    map_height: int,
):
    """
    Draw the UI elements (timer and requirements).

    Args:
        canvas: Pygame surface to draw on
        time_remaining: Steps remaining in the round
        time_limit: Total time limit for the round
        requirements: List of requirement dictionaries
        map_width: Width of the map
        map_height: Height of the map
    """
    # Draw timer bar at the top
    bar_height = 30
    bar_width = map_width - 40
    bar_x = 20
    bar_y = 10

    # Background
    pygame.draw.rect(canvas, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))

    # Progress
    progress = time_remaining / time_limit if time_limit > 0 else 0
    progress_width = int(bar_width * progress)
    color = (100, 255, 100) if progress > 0.5 else (255, 200, 100) if progress > 0.25 else (255, 100, 100)
    pygame.draw.rect(canvas, color, (bar_x, bar_y, progress_width, bar_height))

    # Border
    pygame.draw.rect(canvas, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), 3)

    # Time text
    font = pygame.font.SysFont("Arial", 14)
    time_text = f"{time_remaining} / {time_limit}"
    text_surface = font.render(time_text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
    canvas.blit(text_surface, text_rect)

    # Draw requirements at the bottom
    req_y = map_height - 60
    req_x = 20

    for i, req in enumerate(requirements):
        # Draw requirement box
        box_size = 40
        box_x = req_x + i * (box_size + 10)

        # Background color based on completion
        bg_color = (100, 255, 100) if req.get("completed", False) else (150, 150, 150)
        pygame.draw.rect(canvas, bg_color, (box_x, req_y, box_size, box_size))
        pygame.draw.rect(canvas, (50, 50, 50), (box_x, req_y, box_size, box_size), 2)

        essence_types = req["base_essences"]
        if len(essence_types) == 1:
            color = ESSENCE_TYPES[essence_types[0]][1]
            center = (box_x + box_size // 2, req_y + box_size // 2)
            pygame.draw.circle(canvas, color, center, box_size // 3)
        else:
            n_parts = len(essence_types)
            segment_width = box_size // n_parts
            for j, etype in enumerate(essence_types):
                color = ESSENCE_TYPES[etype][1]
                seg_rect = pygame.Rect(box_x + j * segment_width, req_y, segment_width, box_size)
                pygame.draw.rect(canvas, color, seg_rect)


# ============================================================================
# Round Management
# ============================================================================


class RoundManager:
    """Manages rounds and their configurations."""

    def __init__(self, rounds_config: list[dict]):
        """
        Initialize with a list of round configurations.

        Args:
            rounds_config: List of round dictionaries
        """
        self.rounds = rounds_config
        self.current_round_index = 0

    def get_current_round(self) -> dict | None:
        """Get the current round configuration."""
        if self.current_round_index < len(self.rounds):
            return self.rounds[self.current_round_index]
        return None

    def advance_round(self):
        """Move to the next round."""
        self.current_round_index += 1

    def reset(self):
        """Reset to the first round."""
        self.current_round_index = 0

    def is_complete(self) -> bool:
        """Check if all rounds are complete."""
        return self.current_round_index >= len(self.rounds)

    @staticmethod
    def load_from_file(filepath: str) -> "RoundManager":
        """Load rounds from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        rounds = data.get("rounds", [])
        return RoundManager(rounds)

    @staticmethod
    def create_default_rounds() -> "RoundManager":
        """Load default rounds from rounds.json file."""
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rounds_file = os.path.join(current_dir, "rounds.json")

        return RoundManager.load_from_file(rounds_file)


# ============================================================================
# Layout Management
# ============================================================================


def create_default_layout(
    space: pymunk.Space,
    tile_size: float,
    layout_config: dict,
    tool_params: dict,
    dispenser_params: dict,
) -> dict[str, Any]:
    """
    Create the laboratory layout from supplied configuration (positions and tool parameters).

    Args:
        space: Pymunk physics space
        tile_size: Size of each tile in pixels
        layout_config: Dict with dispenser/tool/player positions and essence types
        tool_params: Dict with per-tool size/colors/timings
        dispenser_params: Dict with dispenser size/timing config

    Returns a dictionary containing all game objects.
    """
    # Create dispensers from layout config
    dispensers = []
    for dispenser_data in layout_config["dispensers"]:
        dispenser = Dispenser(
            space,
            tuple(dispenser_data["position"]),
            dispenser_data["essence_type"],
            tile_size,
            size_multiplier=dispenser_params["size_multiplier"],
            cooldown_duration=dispenser_params["cooldown_duration"],
        )
        dispensers.append(dispenser)

    tools_data = layout_config["tools"]
    enchanter = Enchanter(
        space,
        tuple(tools_data["enchanter"]),
        tile_size,
        size_multiplier=tool_params["enchanter"]["size_multiplier"],
        color=tool_params["enchanter"]["color"],
        processing_time=tool_params["enchanter"]["processing_time"],
        eject_delay=tool_params["enchanter"]["eject_delay"],
    )
    refiner = Refiner(
        space,
        tuple(tools_data["refiner"]),
        tile_size,
        size_multiplier=tool_params["refiner"]["size_multiplier"],
        color=tool_params["refiner"]["color"],
        processing_time=tool_params["refiner"]["processing_time"],
        eject_delay=tool_params["refiner"]["eject_delay"],
    )
    cauldron = Cauldron(
        space,
        tuple(tools_data["cauldron"]),
        tile_size,
        size_multiplier=tool_params["cauldron"]["size_multiplier"],
        color=tool_params["cauldron"]["color"],
        stir_time=tool_params["cauldron"]["stir_time"],
        eject_delay=tool_params["cauldron"]["eject_delay"],
    )
    bottler = Bottler(
        space,
        tuple(tools_data["bottler"]),
        tile_size,
        size_multiplier=tool_params["bottler"]["size_multiplier"],
        color=tool_params["bottler"]["color"],
        bottling_time=tool_params["bottler"]["bottling_time"],
        eject_delay=tool_params["bottler"]["eject_delay"],
    )
    trash_can = TrashCan(
        space,
        tuple(tools_data["trash_can"]),
        tile_size,
        size_multiplier=tool_params["trash_can"]["size_multiplier"],
        color=tool_params["trash_can"]["color"],
    )
    delivery_window = DeliveryWindow(
        space,
        tuple(tools_data["delivery_window"]),
        tile_size,
        width_multiplier=tool_params["delivery_window"]["width_multiplier"],
        height_multiplier=tool_params["delivery_window"]["height_multiplier"],
        color=tool_params["delivery_window"]["color"],
        feedback_duration=tool_params["delivery_window"]["feedback_duration"],
    )

    return {
        "dispensers": dispensers,
        "enchanter": enchanter,
        "refiner": refiner,
        "cauldron": cauldron,
        "bottler": bottler,
        "trash_can": trash_can,
        "delivery_window": delivery_window,
    }
