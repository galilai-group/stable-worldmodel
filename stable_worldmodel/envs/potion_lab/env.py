"""
Potion Lab environment module.
"""

import os
from collections import Counter
from collections.abc import Sequence
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

import stable_worldmodel as swm

from .entities import (
    ESSENCE_TYPES,
    Bottler,
    Cauldron,
    DeliveryWindow,
    Dispenser,
    Enchanter,
    Essence,
    EssenceState,
    PhysicsConfig,
    Player,
    Refiner,
    TrashCan,
)
from .game_logic import (
    CollisionHandler,
    RoundManager,
    add_walls,
    create_default_layout,
    draw_dispenser,
    draw_essence,
    draw_player,
    draw_tool,
    draw_ui,
    render_essence,
    setup_physics_space,
)


DEFAULT_VARIATIONS = (
    "player.color",
    "player.mass",
    "player.friction",
    "player.elasticity",
    "essence.mass",
    "essence.friction",
    "essence.elasticity",
)


class PotionLab(gym.Env):
    """
    Potion Lab - A continual learning environment for world models.

    The agent must learn to brew potions by:
    1. Collecting essences from dispensers
    2. Processing them through various tools (Enchanter, Refiner, Cauldron, Bottler)
    3. Delivering completed potions to the delivery window

    Features:
    - Physics-based interactions using Pymunk
    - Progressive difficulty through 60+ rounds
    - Compositional generalization testing
    - Visual patterns for essence states (stripes=enchanted, dots=refined)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        resolution: int = 512,
        render_action: bool = False,
        rounds_path: str | None = None,
    ):
        """
        Initialize the Potion Lab environment.

        Args:
            render_mode: "human" or "rgb_array"
            resolution: Render resolution (square image)
            render_action: Whether to render action indicators
            rounds_path: Path to rounds JSON file (default: potion_lab/rounds.json)
        """
        super().__init__()

        self._seed = None
        self.render_mode = render_mode
        self.render_size = resolution
        self.window_size = 512
        self.render_action = render_action

        # Config files (use defaults if not provided)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.rounds_path = rounds_path or os.path.join(current_dir, "rounds.json")

        # Initialize spatial parameters from defaults
        self.ui_top_height = 50
        self.ui_bottom_height = 80

        # Lab fills remaining space between UI bars
        self.map_width_tiles = 16
        self.map_height_tiles = 16
        self.tile_size = self.window_size / self.map_width_tiles  # Updated on reset
        self.map_width = self.window_size
        self.map_height = self.window_size - self.ui_top_height - self.ui_bottom_height

        # Action space: cursor position (x, y) in world coordinates
        self.action_space = spaces.Box(low=0.0, high=float(self.window_size), shape=(2,), dtype=np.float32)

        # Proprio vector structure (75 floats total):
        # [0-1]   player x, y
        # [2-3]   player vx, vy
        # [4]     time_remaining_normalized (0-1)
        # [5-74]  requirements (5 slots × 14 floats each)
        #         Per requirement: [essence_types(4), enchanted(4), refined(4), bottled(1), completed(1)]
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8),
                "proprio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(75,),
                    dtype=np.float32,
                ),
            }
        )

        self.variation_space = swm.spaces.Dict(
            {
                "player": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array(pygame.Color("RoyalBlue")[:3], dtype=np.uint8)
                        ),  # color is used when asset is not available
                        "size": swm.spaces.Box(
                            low=8.0,
                            high=50.0,
                            init_value=32.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "hitbox_size": swm.spaces.Box(
                            low=8.0,
                            high=50.0,
                            init_value=20.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "mass": swm.spaces.Box(
                            low=0.5,
                            high=3.0,
                            init_value=1.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "friction": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.2,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "elasticity": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "speed": swm.spaces.Box(
                            low=100.0,
                            high=400.0,
                            init_value=200.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "force": swm.spaces.Box(
                            low=10.0,
                            high=140.0,
                            init_value=70.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "start_position": swm.spaces.Box(
                            low=50.0,
                            high=self.window_size - 50.0,
                            init_value=np.array([self.window_size / 2, self.window_size / 2], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "player_control": swm.spaces.Dict(
                    {
                        "distance_slowdown_scale": swm.spaces.Box(
                            low=1.0, high=200.0, init_value=20.0, shape=(), dtype=np.float32
                        ),
                    }
                ),
                "essence": swm.spaces.Dict(
                    {
                        "mass": swm.spaces.Box(
                            low=0.1,
                            high=2.0,
                            init_value=1.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "friction": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=1.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "elasticity": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=1.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "radius_scale": swm.spaces.Box(
                            low=0.1,
                            high=1.0,
                            init_value=0.35,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "drag_coefficient": swm.spaces.Box(
                            low=0.0,
                            high=10.0,
                            init_value=5.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "physics": swm.spaces.Dict(
                    {
                        "gravity": swm.spaces.Box(
                            low=-10.0,
                            high=10.0,
                            init_value=np.array([0.0, 0.0], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "background": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(init_value=np.array([100, 0, 0], dtype=np.uint8)),
                    }
                ),
                "grid": swm.spaces.Dict(
                    {
                        "map_width_tiles": swm.spaces.Box(
                            low=8.0,
                            high=64.0,
                            init_value=float(self.map_width_tiles),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "map_height_tiles": swm.spaces.Box(
                            low=8.0,
                            high=64.0,
                            init_value=float(self.map_height_tiles),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "ui": swm.spaces.Dict(
                    {
                        "top_height": swm.spaces.Box(
                            low=0.0,
                            high=200.0,
                            init_value=float(self.ui_top_height),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "bottom_height": swm.spaces.Box(
                            low=0.0,
                            high=200.0,
                            init_value=float(self.ui_bottom_height),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "layout": swm.spaces.Dict(
                    {
                        "tools": swm.spaces.Dict(
                            {
                                "enchanter_position": swm.spaces.Box(
                                    low=0.0,
                                    high=float(self.window_size),
                                    init_value=np.array([100.0, 148.0], dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                                "refiner_position": swm.spaces.Box(
                                    low=0.0,
                                    high=float(self.window_size),
                                    init_value=np.array([412.0, 148.0], dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                                "cauldron_position": swm.spaces.Box(
                                    low=0.0,
                                    high=float(self.window_size),
                                    init_value=np.array([256.0, 232.0], dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                                "bottler_position": swm.spaces.Box(
                                    low=0.0,
                                    high=float(self.window_size),
                                    init_value=np.array([100.0, 300.0], dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                                "trash_can_position": swm.spaces.Box(
                                    low=0.0,
                                    high=float(self.window_size),
                                    init_value=np.array([400.0, 300.0], dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                                "delivery_window_position": swm.spaces.Box(
                                    low=0.0,
                                    high=float(self.window_size),
                                    init_value=np.array([256.0, 350.0], dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                            }
                        ),
                        "dispensers": swm.spaces.Dict(
                            {
                                "d1_position": swm.spaces.Box(
                                    low=0.0,
                                    high=float(self.window_size),
                                    init_value=np.array([181.0, 48.0], dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                                "d1_type": swm.spaces.Box(
                                    low=1.0,
                                    high=8.0,
                                    init_value=1.0,
                                    shape=(),
                                    dtype=np.float32,
                                ),
                                "d2_position": swm.spaces.Box(
                                    low=0.0,
                                    high=float(self.window_size),
                                    init_value=np.array([231.0, 48.0], dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                                "d2_type": swm.spaces.Box(
                                    low=1.0,
                                    high=8.0,
                                    init_value=2.0,
                                    shape=(),
                                    dtype=np.float32,
                                ),
                                "d3_position": swm.spaces.Box(
                                    low=0.0,
                                    high=float(self.window_size),
                                    init_value=np.array([281.0, 48.0], dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                                "d3_type": swm.spaces.Box(
                                    low=1.0,
                                    high=8.0,
                                    init_value=3.0,
                                    shape=(),
                                    dtype=np.float32,
                                ),
                                "d4_position": swm.spaces.Box(
                                    low=0.0,
                                    high=float(self.window_size),
                                    init_value=np.array([331.0, 48.0], dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                                "d4_type": swm.spaces.Box(
                                    low=1.0,
                                    high=8.0,
                                    init_value=4.0,
                                    shape=(),
                                    dtype=np.float32,
                                ),
                            }
                        ),
                    }
                ),
                "tools_config": swm.spaces.Dict(
                    {
                        "eject_offset_multiplier": swm.spaces.Box(
                            low=0.0, high=4.0, init_value=2, shape=(), dtype=np.float32
                        ),
                        "enchanter": swm.spaces.Dict(
                            {
                                "size_multiplier": swm.spaces.Box(
                                    low=0.5, high=3.0, init_value=1.2, shape=(), dtype=np.float32
                                ),
                                "color": swm.spaces.RGBBox(init_value=np.array([138, 43, 226], dtype=np.uint8)),
                                "processing_time": swm.spaces.Box(
                                    low=1.0, high=600.0, init_value=120.0, shape=(), dtype=np.float32
                                ),
                                "eject_delay": swm.spaces.Box(
                                    low=0.0, high=120.0, init_value=6.0, shape=(), dtype=np.float32
                                ),
                            }
                        ),
                        "refiner": swm.spaces.Dict(
                            {
                                "size_multiplier": swm.spaces.Box(
                                    low=0.5, high=3.0, init_value=1.2, shape=(), dtype=np.float32
                                ),
                                "color": swm.spaces.RGBBox(init_value=np.array([184, 134, 11], dtype=np.uint8)),
                                "processing_time": swm.spaces.Box(
                                    low=1.0, high=600.0, init_value=150.0, shape=(), dtype=np.float32
                                ),
                                "eject_delay": swm.spaces.Box(
                                    low=0.0, high=120.0, init_value=6.0, shape=(), dtype=np.float32
                                ),
                            }
                        ),
                        "cauldron": swm.spaces.Dict(
                            {
                                "size_multiplier": swm.spaces.Box(
                                    low=0.5, high=3.0, init_value=1.5, shape=(), dtype=np.float32
                                ),
                                "color": swm.spaces.RGBBox(init_value=np.array([47, 79, 79], dtype=np.uint8)),
                                "stir_time": swm.spaces.Box(
                                    low=1.0, high=600.0, init_value=60.0, shape=(), dtype=np.float32
                                ),
                                "eject_delay": swm.spaces.Box(
                                    low=0.0, high=120.0, init_value=6.0, shape=(), dtype=np.float32
                                ),
                            }
                        ),
                        "bottler": swm.spaces.Dict(
                            {
                                "size_multiplier": swm.spaces.Box(
                                    low=0.5, high=3.0, init_value=1.0, shape=(), dtype=np.float32
                                ),
                                "color": swm.spaces.RGBBox(init_value=np.array([176, 196, 222], dtype=np.uint8)),
                                "bottling_time": swm.spaces.Box(
                                    low=1.0, high=600.0, init_value=90.0, shape=(), dtype=np.float32
                                ),
                                "eject_delay": swm.spaces.Box(
                                    low=0.0, high=120.0, init_value=6.0, shape=(), dtype=np.float32
                                ),
                            }
                        ),
                        "trash_can": swm.spaces.Dict(
                            {
                                "size_multiplier": swm.spaces.Box(
                                    low=0.2, high=3.0, init_value=0.8, shape=(), dtype=np.float32
                                ),
                                "color": swm.spaces.RGBBox(init_value=np.array([105, 105, 105], dtype=np.uint8)),
                            }
                        ),
                        "delivery_window": swm.spaces.Dict(
                            {
                                "width_multiplier": swm.spaces.Box(
                                    low=0.5, high=3.0, init_value=3, shape=(), dtype=np.float32
                                ),
                                "height_multiplier": swm.spaces.Box(
                                    low=0.5, high=3.0, init_value=2.0, shape=(), dtype=np.float32
                                ),
                                "color": swm.spaces.RGBBox(init_value=np.array([60, 179, 113], dtype=np.uint8)),
                                "feedback_duration": swm.spaces.Box(
                                    low=1.0, high=120.0, init_value=12.0, shape=(), dtype=np.float32
                                ),
                            }
                        ),
                    }
                ),
                "dispenser_config": swm.spaces.Dict(
                    {
                        "size_multiplier": swm.spaces.Box(
                            low=0.2, high=3.0, init_value=0.8, shape=(), dtype=np.float32
                        ),
                        "cooldown_duration": swm.spaces.Box(
                            low=1.0, high=300.0, init_value=30.0, shape=(), dtype=np.float32
                        ),
                        "spawn_offset_multiplier": swm.spaces.Box(
                            low=0.0, high=4.0, init_value=1.25, shape=(), dtype=np.float32
                        ),
                    }
                ),
                "environment": swm.spaces.Dict(
                    {
                        "wall_thickness": swm.spaces.Box(
                            low=1.0, high=50.0, init_value=10.0, shape=(), dtype=np.float32
                        ),
                        "window_size": swm.spaces.Box(
                            low=256.0,
                            high=1024.0,
                            init_value=float(self.window_size),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "rewards": swm.spaces.Dict(
                    {
                        "substep_reward": swm.spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.1,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "round_complete_bonus": swm.spaces.Box(
                            low=0.0,
                            high=10.0,
                            init_value=1.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "timeout_penalty": swm.spaces.Box(
                            low=-10.0,
                            high=0.0,
                            init_value=-1.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
            },
            sampling_order=[
                "environment",
                "background",
                "physics",
                "essence",
                "player",
                "ui",
                "layout",
                "tools_config",
                "dispenser_config",
                "player_control",
                "grid",
                "rewards",
            ],
        )

        # Initialize derived config using default variation values
        self._apply_variation_config()

        # Pygame and rendering
        self.window = None
        self.clock = None
        self.canvas = None

        # Game state
        self.space = None
        self.player = None
        self.essences = []
        self.tools = {}
        self.dispensers = []
        self.collision_handler = None
        self.round_manager = None

        # Unlocked items tracking
        self.unlocked_dispensers = set()  # Set of essence_types that are unlocked
        self.unlocked_tools = set()  # Set of tool names that are unlocked

        # Timing
        self.step_count = 0
        self.round_time_remaining = 0

        self.latest_action = None

        # Dense reward system - quotas and pending rewards
        self.quotas = self._empty_quotas()
        self.pending_reward = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

        options = options or {}

        self.unlocked_dispensers = set()
        self.unlocked_tools = set()

        self.variation_space.reset()

        variations = options.get("variation", DEFAULT_VARIATIONS)

        if not isinstance(variations, Sequence):
            raise ValueError("variation option must be a Sequence containing variation names to sample")

        self.variation_space.update(variations)

        assert self.variation_space.check(debug=True), "Variation values must be within variation space!"

        # Refresh derived configuration from sampled variations
        self._apply_variation_config()

        self._setup()

        self.round_manager = RoundManager.load_from_file(self.rounds_path)

        round_config = self.round_manager.get_current_round()
        if round_config is None:
            raise RuntimeError("No rounds configured!")

        # Update unlocked items for this round
        self._update_unlocked_items(round_config)
        self._update_enabled_items()

        requirements = round_config["required_items"].copy()
        for req in requirements:
            req["completed"] = False
        self.tools["delivery_window"].set_requirements(requirements)

        self.round_time_remaining = round_config["time_limit"]
        self.step_count = 0

        # Set background color for this round if specified
        self._set_round_background_color(round_config)

        # Calculate quotas for dense rewards
        self.quotas = self._calculate_quotas(requirements)
        self.pending_reward = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _apply_variation_config(self):
        """Apply sampled variation values to derived configuration and cached params."""
        self.window_size = float(self.variation_space["environment"]["window_size"].value)
        self.map_width_tiles = int(self.variation_space["grid"]["map_width_tiles"].value)
        self.map_height_tiles = int(self.variation_space["grid"]["map_height_tiles"].value)

        self.ui_top_height = float(self.variation_space["ui"]["top_height"].value)
        self.ui_bottom_height = float(self.variation_space["ui"]["bottom_height"].value)

        self.map_width = self.window_size
        self.map_height = self.window_size - self.ui_top_height - self.ui_bottom_height
        self.tile_size = self.map_width / max(1, self.map_width_tiles)

        # Update action space to match current window size
        self.action_space = spaces.Box(
            low=0.0,
            high=float(self.window_size),
            shape=(2,),
            dtype=np.float32,
        )

        # Layout configuration
        layout_disp = self.variation_space["layout"]["dispensers"]
        layout_tools = self.variation_space["layout"]["tools"]
        self.layout_config = {
            "dispensers": [
                {
                    "position": layout_disp["d1_position"].value.tolist(),
                    "essence_type": int(layout_disp["d1_type"].value),
                },
                {
                    "position": layout_disp["d2_position"].value.tolist(),
                    "essence_type": int(layout_disp["d2_type"].value),
                },
                {
                    "position": layout_disp["d3_position"].value.tolist(),
                    "essence_type": int(layout_disp["d3_type"].value),
                },
                {
                    "position": layout_disp["d4_position"].value.tolist(),
                    "essence_type": int(layout_disp["d4_type"].value),
                },
            ],
            "tools": {
                "enchanter": layout_tools["enchanter_position"].value.tolist(),
                "refiner": layout_tools["refiner_position"].value.tolist(),
                "cauldron": layout_tools["cauldron_position"].value.tolist(),
                "bottler": layout_tools["bottler_position"].value.tolist(),
                "trash_can": layout_tools["trash_can_position"].value.tolist(),
                "delivery_window": layout_tools["delivery_window_position"].value.tolist(),
            },
        }

        def _to_color(arr):
            return tuple(int(x) for x in arr.tolist())

        tool_cfg = self.variation_space["tools_config"]
        self.tool_params = {
            "eject_offset_multiplier": float(tool_cfg["eject_offset_multiplier"].value),
            "enchanter": {
                "size_multiplier": float(tool_cfg["enchanter"]["size_multiplier"].value),
                "color": _to_color(tool_cfg["enchanter"]["color"].value),
                "processing_time": int(tool_cfg["enchanter"]["processing_time"].value),
                "eject_delay": int(tool_cfg["enchanter"]["eject_delay"].value),
            },
            "refiner": {
                "size_multiplier": float(tool_cfg["refiner"]["size_multiplier"].value),
                "color": _to_color(tool_cfg["refiner"]["color"].value),
                "processing_time": int(tool_cfg["refiner"]["processing_time"].value),
                "eject_delay": int(tool_cfg["refiner"]["eject_delay"].value),
            },
            "cauldron": {
                "size_multiplier": float(tool_cfg["cauldron"]["size_multiplier"].value),
                "color": _to_color(tool_cfg["cauldron"]["color"].value),
                "stir_time": int(tool_cfg["cauldron"]["stir_time"].value),
                "eject_delay": int(tool_cfg["cauldron"]["eject_delay"].value),
            },
            "bottler": {
                "size_multiplier": float(tool_cfg["bottler"]["size_multiplier"].value),
                "color": _to_color(tool_cfg["bottler"]["color"].value),
                "bottling_time": int(tool_cfg["bottler"]["bottling_time"].value),
                "eject_delay": int(tool_cfg["bottler"]["eject_delay"].value),
            },
            "trash_can": {
                "size_multiplier": float(tool_cfg["trash_can"]["size_multiplier"].value),
                "color": _to_color(tool_cfg["trash_can"]["color"].value),
            },
            "delivery_window": {
                "width_multiplier": float(tool_cfg["delivery_window"]["width_multiplier"].value),
                "height_multiplier": float(tool_cfg["delivery_window"]["height_multiplier"].value),
                "color": _to_color(tool_cfg["delivery_window"]["color"].value),
                "feedback_duration": int(tool_cfg["delivery_window"]["feedback_duration"].value),
            },
        }

        dispenser_cfg = self.variation_space["dispenser_config"]
        self.dispenser_params = {
            "size_multiplier": float(dispenser_cfg["size_multiplier"].value),
            "cooldown_duration": int(dispenser_cfg["cooldown_duration"].value),
            "spawn_offset_multiplier": float(dispenser_cfg["spawn_offset_multiplier"].value),
        }

        essence_cfg = self.variation_space["essence"]
        self.essence_config = {
            "radius_scale": float(essence_cfg["radius_scale"].value),
            "drag_coefficient": float(essence_cfg["drag_coefficient"].value),
        }

        player_control_cfg = self.variation_space["player_control"]
        self.player_control_config = {
            "distance_slowdown_scale": float(player_control_cfg["distance_slowdown_scale"].value),
        }

        env_cfg = self.variation_space["environment"]
        self.env_config = {"wall_thickness": float(env_cfg["wall_thickness"].value)}

        rewards_cfg = self.variation_space["rewards"]
        self.reward_config = {
            "substep_reward": float(rewards_cfg["substep_reward"].value),
            "round_complete_bonus": float(rewards_cfg["round_complete_bonus"].value),
            "timeout_penalty": float(rewards_cfg["timeout_penalty"].value),
        }

        # Clamp player start position to the playable map (accounts for UI space)
        start_pos = self.variation_space["player"]["start_position"].value
        clamped_start = np.array(
            [
                np.clip(start_pos[0], self.tile_size, self.map_width - self.tile_size),
                np.clip(start_pos[1], self.tile_size, self.map_height - self.tile_size),
            ],
            dtype=np.float32,
        )
        # Update the variation space value
        self.variation_space["player"]["start_position"]._value = clamped_start

    def _empty_quotas(self) -> dict:
        """Return an empty quota structure."""
        return {
            "dispense": Counter(),  # {essence_type: count}
            "enchant": Counter(),  # {essence_type: count}
            "refine": Counter(),  # {essence_type: count}
            "combine": Counter(),  # {(types_tuple, ench_tuple, ref_tuple): count}
            "bottle": Counter(),  # {(types_tuple, ench_tuple, ref_tuple, is_combined): count}
        }

    def _calculate_quotas(self, requirements: list[dict]) -> dict:
        """
        Calculate quotas for dense rewards based on round requirements.

        Args:
            requirements: List of recipe requirement dictionaries

        Returns:
            Dictionary with quotas for each reward event type
        """
        quotas = self._empty_quotas()

        for req in requirements:
            types = req["base_essences"]
            enchanted = req.get("enchanted", [False] * len(types))
            refined = req.get("refined", [False] * len(types))
            bottled = req.get("bottled", False)

            # Dispense quota: count each essence type needed
            for t in types:
                quotas["dispense"][t] += 1

            # Enchant/Refine quota: count per-component property requirements
            for i, t in enumerate(types):
                if enchanted[i]:
                    quotas["enchant"][t] += 1
                if refined[i]:
                    quotas["refine"][t] += 1

            # Combine quota: only if multi-essence, exact state required
            if len(types) > 1:
                combo_key = (
                    tuple(sorted(types)),
                    tuple(enchanted),
                    tuple(refined),
                )
                quotas["combine"][combo_key] += 1

            # Bottle quota: exact pre-bottle state
            if bottled:
                pre_bottle_key = (
                    tuple(sorted(types)),
                    tuple(enchanted),
                    tuple(refined),
                    len(types) > 1,  # is_combined
                )
                quotas["bottle"][pre_bottle_key] += 1

        return quotas

    def _setup(self):
        """Initialize physics space and game objects."""
        self.space = setup_physics_space()

        self.space.gravity = self.variation_space["physics"]["gravity"].value.tolist()
        self.space.damping = 1.0

        add_walls(self.space, self.map_width, self.map_height, wall_thickness=self.env_config["wall_thickness"])

        layout = create_default_layout(
            self.space,
            self.tile_size,
            self.layout_config,
            self.tool_params,
            self.dispenser_params,
        )

        # Create player with physics properties from variation space
        start_pos = self.variation_space["player"]["start_position"].value
        self.player = Player(
            self.space,
            position=(float(start_pos[0]), float(start_pos[1])),
            tile_size=self.tile_size,
            render_size=self.variation_space["player"]["size"].value,
            hitbox_size=self.variation_space["player"]["hitbox_size"].value,
            mass=self.variation_space["player"]["mass"].value,
            friction=self.variation_space["player"]["friction"].value,
            elasticity=self.variation_space["player"]["elasticity"].value,
            max_velocity=self.variation_space["player"]["speed"].value,
            force_scale=self.variation_space["player"]["force"].value,
            color=tuple(self.variation_space["player"]["color"].value.tolist()),
            distance_slowdown_scale=self.player_control_config["distance_slowdown_scale"],
        )

        self.tools = {
            "enchanter": layout["enchanter"],
            "refiner": layout["refiner"],
            "cauldron": layout["cauldron"],
            "bottler": layout["bottler"],
            "trash_can": layout["trash_can"],
            "delivery_window": layout["delivery_window"],
        }
        self.dispensers = layout["dispensers"]

        # Initially disable all tools and dispensers (they start removed from physics space)
        # They will be enabled later when rounds require them
        for tool in self.tools.values():
            tool.disable()
        for dispenser in self.dispensers:
            dispenser.disable()

        self.collision_handler = CollisionHandler(self)
        self.collision_handler.setup_handlers(self.space)

        self.essences = []

    def step(self, action: np.ndarray):
        """
        Execute one step in the environment.

        Args:
            action: [x, y] cursor position in world coordinates [0, window_size]

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.latest_action = action

        self.player.apply_action(action)

        self.space.step(PhysicsConfig.TIMESTEP)

        self.collision_handler.update()

        for tool in self.tools.values():
            tool.update(PhysicsConfig.TIMESTEP)

        for dispenser in self.dispensers:
            dispenser.update(PhysicsConfig.TIMESTEP)

        self._check_tool_ejections()

        self.round_time_remaining -= 1
        self.step_count += 1

        terminated = False
        truncated = False

        # Collect accumulated substep rewards
        reward = self.pending_reward
        self.pending_reward = 0.0

        if self.tools["delivery_window"].all_requirements_met():
            # Round is complete! Add round completion bonus
            reward += self.reward_config["round_complete_bonus"]
            self.round_manager.advance_round()

            if self.round_manager.is_complete():
                terminated = True
            else:
                self._load_next_round()

        elif self.round_time_remaining <= 0:
            # Round failed. Add timeout penalty.
            reward += self.reward_config["timeout_penalty"]
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _load_next_round(self):
        """Load the next round configuration without full reset."""
        round_config = self.round_manager.get_current_round()
        if round_config is None:
            return

        # Update unlocked items for the new round
        self._update_unlocked_items(round_config)
        self._update_enabled_items()

        for essence in self.essences[:]:
            essence.remove_from_world()
        self.essences.clear()

        for tool in self.tools.values():
            tool.reset()

        start_pos = self.variation_space["player"]["start_position"].value
        self.player.body.position = (float(start_pos[0]), float(start_pos[1]))
        self.player.body.velocity = (0, 0)

        if hasattr(self, "collision_handler"):
            self.collision_handler.player_stirring_cauldron = False

        requirements = round_config["required_items"].copy()
        for req in requirements:
            req["completed"] = False
        self.tools["delivery_window"].set_requirements(requirements)

        self.round_time_remaining = round_config["time_limit"]

        # Set background color for this round if specified
        self._set_round_background_color(round_config)

        # Recalculate quotas for new round
        self.quotas = self._calculate_quotas(requirements)
        self.pending_reward = 0.0

    def _set_round_background_color(self, round_config: dict):
        """Set background color for the current round if specified in round config."""
        if "background_color" in round_config:
            bg_color = np.array(round_config["background_color"], dtype=np.uint8)
            self.variation_space["background"]["color"]._value = bg_color

    def _get_round_requirements(self, round_config: dict) -> tuple[set[int], set[str]]:
        """
        Analyze a round's requirements to determine needed dispensers and tools.

        Returns:
            tuple: (essence_types_needed, tools_needed)
        """
        essence_types_needed = set()
        tools_needed = set()

        for req in round_config.get("required_items", []):
            # Check essence types needed
            base_essences = req.get("base_essences", [])
            essence_types_needed.update(base_essences)

            # Check tools needed based on processing requirements
            enchanted = req.get("enchanted", [False] * len(base_essences))
            refined = req.get("refined", [False] * len(base_essences))
            bottled = req.get("bottled", False)

            if any(enchanted):
                tools_needed.add("enchanter")
            if any(refined):
                tools_needed.add("refiner")
            if bottled:
                tools_needed.add("bottler")
            # Cauldron is needed for mixing/combining essences (when multiple essence types in one item)
            if len(base_essences) > 1:
                tools_needed.add("cauldron")

        return essence_types_needed, tools_needed

    def _update_enabled_items(self):
        """Update the enabled/disabled state of tools and dispensers based on unlocked items."""
        # Update dispensers
        for dispenser in self.dispensers:
            if dispenser.essence_type in self.unlocked_dispensers:
                dispenser.enable()
            else:
                dispenser.disable()

        # Update tools (delivery window is always enabled)
        for tool_name, tool in self.tools.items():
            if tool_name == "delivery_window":
                tool.enable()  # Always enabled
            elif tool_name in self.unlocked_tools:
                tool.enable()
            else:
                tool.disable()

    def _update_unlocked_items(self, round_config: dict):
        """Update the set of unlocked dispensers and tools based on round requirements."""
        essence_types_needed, tools_needed = self._get_round_requirements(round_config)

        # Unlock new dispensers
        self.unlocked_dispensers.update(essence_types_needed)

        # Unlock new tools
        self.unlocked_tools.update(tools_needed)

    def _check_tool_ejections(self):
        """Check if any tools are ready to eject processed essences and award substep rewards."""
        eject_offset = self.tile_size * self.tool_params["eject_offset_multiplier"]

        # Get essence physics properties from variation space
        mass = self.variation_space["essence"]["mass"].value
        friction = self.variation_space["essence"]["friction"].value
        elasticity = self.variation_space["essence"]["elasticity"].value
        radius_scale = self.essence_config["radius_scale"]
        drag_coefficient = self.essence_config["drag_coefficient"]

        # Define eject offsets for each tool (x_offset, y_offset)
        tool_offsets = {
            "enchanter": (eject_offset, 0),
            "refiner": (-eject_offset, 0),
            "cauldron": (0, eject_offset),
            "bottler": (0, eject_offset),
        }

        substep_reward = self.reward_config["substep_reward"]

        # Handle Enchanter ejection
        enchanter = self.tools["enchanter"]
        essence_state, components_enchanted = enchanter.eject_essence()
        if essence_state is not None:
            # Award rewards for each component that gained enchantment
            for essence_type in components_enchanted:
                if self.quotas["enchant"][essence_type] > 0:
                    self.pending_reward += substep_reward
                    self.quotas["enchant"][essence_type] -= 1

            pos = (
                enchanter.position[0] + tool_offsets["enchanter"][0],
                enchanter.position[1] + tool_offsets["enchanter"][1],
            )
            self._spawn_essence(pos, essence_state, mass, friction, elasticity, radius_scale, drag_coefficient)

        # Handle Refiner ejection
        refiner = self.tools["refiner"]
        essence_state, components_refined = refiner.eject_essence()
        if essence_state is not None:
            # Award rewards for each component that gained refinement
            for essence_type in components_refined:
                if self.quotas["refine"][essence_type] > 0:
                    self.pending_reward += substep_reward
                    self.quotas["refine"][essence_type] -= 1

            pos = (refiner.position[0] + tool_offsets["refiner"][0], refiner.position[1] + tool_offsets["refiner"][1])
            self._spawn_essence(pos, essence_state, mass, friction, elasticity, radius_scale, drag_coefficient)

        # Handle Cauldron ejection
        cauldron = self.tools["cauldron"]
        essence_state, combo_key = cauldron.eject_essence()
        if essence_state is not None:
            # Award reward if this exact combination is in quota
            if combo_key is not None and self.quotas["combine"][combo_key] > 0:
                self.pending_reward += substep_reward
                self.quotas["combine"][combo_key] -= 1

            pos = (
                cauldron.position[0] + tool_offsets["cauldron"][0],
                cauldron.position[1] + tool_offsets["cauldron"][1],
            )
            self._spawn_essence(pos, essence_state, mass, friction, elasticity, radius_scale, drag_coefficient)

        # Handle Bottler ejection
        bottler = self.tools["bottler"]
        essence_state, pre_bottle_key = bottler.eject_essence()
        if essence_state is not None:
            # Award reward if this exact pre-bottle state is in quota
            if pre_bottle_key is not None and self.quotas["bottle"][pre_bottle_key] > 0:
                self.pending_reward += substep_reward
                self.quotas["bottle"][pre_bottle_key] -= 1

            pos = (bottler.position[0] + tool_offsets["bottler"][0], bottler.position[1] + tool_offsets["bottler"][1])
            self._spawn_essence(pos, essence_state, mass, friction, elasticity, radius_scale, drag_coefficient)

    def _spawn_essence(
        self,
        pos: tuple[float, float],
        essence_state: EssenceState,
        mass: float,
        friction: float,
        elasticity: float,
        radius_scale: float,
        drag_coefficient: float,
    ):
        """Helper to spawn an essence at a position."""
        essence = Essence(
            self.space,
            pos,
            essence_state,
            self.tile_size,
            mass,
            friction,
            elasticity,
            radius_scale=radius_scale,
            drag_coefficient=drag_coefficient,
        )
        self.essences.append(essence)

    def _encode_requirements(self) -> np.ndarray:
        """
        Encode requirements into a fixed-size vector.

        Structure (5 slots x 14 floats):
        Per slot:
        [0-3]   essence_types (4 floats)
        [4-7]   enchanted (4 floats)
        [8-11]  refined (4 floats)
        [12]    bottled (1 float)
        [13]    completed (1 float)
        """
        encoded = np.zeros(5 * 14, dtype=np.float32)
        requirements = self.tools["delivery_window"].required_items

        for i, req in enumerate(requirements[:5]):  # Max 5 requirements
            base_idx = i * 14

            # Essence types (up to 4)
            types = req.get("base_essences", [])
            for j, t in enumerate(types[:4]):
                encoded[base_idx + j] = float(t)

            # Enchanted status
            enchanted = req.get("enchanted", [])
            for j, e in enumerate(enchanted[:4]):
                encoded[base_idx + 4 + j] = 1.0 if e else 0.0

            # Refined status
            refined = req.get("refined", [])
            for j, r in enumerate(refined[:4]):
                encoded[base_idx + 8 + j] = 1.0 if r else 0.0

            # Bottled status
            encoded[base_idx + 12] = 1.0 if req.get("bottled", False) else 0.0

            # Completed status
            encoded[base_idx + 13] = 1.0 if req.get("completed", False) else 0.0

        return encoded

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Get the current observation."""
        img = self.render()

        # Proprioception vector construction

        # 1. Player State (4 floats)
        player_pos = np.array([self.player.body.position.x, self.player.body.position.y], dtype=np.float32)
        player_vel = np.array([self.player.body.velocity.x, self.player.body.velocity.y], dtype=np.float32)

        # 2. Time Remaining (1 float)
        round_config = self.round_manager.get_current_round()
        time_limit = round_config["time_limit"] if round_config else 1.0
        time_norm = np.array([self.round_time_remaining / max(1.0, time_limit)], dtype=np.float32)

        # 3. Requirements (70 floats)
        req_encoded = self._encode_requirements()

        # Combine all parts (75 floats total)
        proprio = np.concatenate([player_pos, player_vel, time_norm, req_encoded])

        return {"image": img, "proprio": proprio}

    def _get_info(self) -> dict[str, Any]:
        """Get auxiliary information."""
        round_config = self.round_manager.get_current_round()

        return {
            "round_index": self.round_manager.current_round_index,
            "time_remaining": self.round_time_remaining,
            "time_limit": round_config["time_limit"] if round_config else 0,
            "requirements_met": self.tools["delivery_window"].all_requirements_met(),
            "num_essences": len(self.essences),
            "player_pos": np.array([self.player.body.position.x, self.player.body.position.y], dtype=np.float32),
            "player_vel": np.array([self.player.body.velocity.x, self.player.body.velocity.y], dtype=np.float32),
            "round_description": round_config.get("description", "") if round_config else "",
            "required_items": self.tools["delivery_window"].required_items,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.canvas is None:
            pygame.init()
            # Canvas is always window_size x window_size (512x512)
            self.canvas = pygame.Surface((self.window_size, self.window_size))

            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Potion Lab")
                self.clock = pygame.time.Clock()

        bg_color = tuple(self.variation_space["background"]["color"].value.tolist())
        ui_bg_color = (max(0, bg_color[0] - 10), max(0, bg_color[1] - 10), max(0, bg_color[2] - 10))

        self.canvas.fill(ui_bg_color)

        game_area_rect = pygame.Rect(0, self.ui_top_height, int(self.map_width), int(self.map_height))
        pygame.draw.rect(self.canvas, bg_color, game_area_rect)

        game_surface = pygame.Surface((int(self.map_width), int(self.map_height)))
        game_surface.fill(bg_color)

        # Only draw unlocked tools (excluding delivery_window which is always shown)
        for tool_name, tool in self.tools.items():
            if tool_name != "delivery_window" and tool_name in self.unlocked_tools:
                draw_tool(game_surface, tool)

        draw_tool(game_surface, self.tools["delivery_window"])

        # Only draw unlocked dispensers
        for dispenser in self.dispensers:
            if dispenser.essence_type in self.unlocked_dispensers:
                draw_dispenser(game_surface, dispenser)

        for essence in self.essences:
            draw_essence(game_surface, essence)

        draw_player(game_surface, self.player)

        self.canvas.blit(game_surface, (0, self.ui_top_height))

        round_config = self.round_manager.get_current_round()
        if round_config:
            self._draw_timer_bar(
                self.round_time_remaining, round_config["time_limit"], int(self.map_width), self.ui_top_height
            )

            self._draw_requirements(
                self.tools["delivery_window"].required_items,
                int(self.map_width),
                self.ui_top_height + int(self.map_height),
                self.ui_bottom_height,
            )

        # Handle human mode
        if self.render_mode == "human":
            self.window.blit(self.canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            return None

        img = np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

        if img.shape[0] != self.render_size or img.shape[1] != self.render_size:
            img = cv2.resize(img, (self.render_size, self.render_size))

        return img

    def _draw_timer_bar(self, time_remaining: int, time_limit: int, width: int, height: int):
        """Draw timer bar at the top."""
        bar_margin = 10
        bar_height = 30
        bar_width = width - 2 * bar_margin
        bar_x = bar_margin
        bar_y = (height - bar_height) // 2

        pygame.draw.rect(self.canvas, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))

        # Progress
        progress = time_remaining / time_limit if time_limit > 0 else 0
        progress_width = int(bar_width * progress)
        color = (100, 255, 100) if progress > 0.5 else (255, 200, 100) if progress > 0.25 else (255, 100, 100)
        pygame.draw.rect(self.canvas, color, (bar_x, bar_y, progress_width, bar_height))

        # Border
        pygame.draw.rect(self.canvas, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), 3)

        # Time text
        font = pygame.font.SysFont("Arial", 14)
        time_text = f"{time_remaining} / {time_limit}"
        text_surface = font.render(time_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
        self.canvas.blit(text_surface, text_rect)

    def _draw_requirements(self, requirements: list[dict], width: int, y_offset: int, height: int):
        """Draw requirements at the bottom - matching floor essence display exactly."""
        margin = 10
        essence_radius = int(0.35 * self.tile_size)
        box_size = essence_radius * 3
        spacing = 10
        start_x = margin
        start_y = y_offset + (height - box_size) // 2

        for i, req in enumerate(requirements):
            box_x = start_x + i * (box_size + spacing)

            # Background color based on completion
            bg_color = (100, 255, 100) if req.get("completed", False) else (80, 80, 85)
            pygame.draw.rect(self.canvas, bg_color, (box_x, start_y, box_size, box_size))
            pygame.draw.rect(self.canvas, (50, 50, 50), (box_x, start_y, box_size, box_size), 2)

            # Draw essence
            essence_types = req["base_essences"]
            enchanted = req.get("enchanted", [False] * len(essence_types))
            refined = req.get("refined", [False] * len(essence_types))
            is_bottled = req.get("bottled", False)

            center = (box_x + box_size // 2, start_y + box_size // 2)

            # Draw essence using unified render function
            render_essence(
                self.canvas,
                center,
                essence_radius,
                essence_types,
                enchanted,
                refined,
                is_bottled,
            )

    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        self.canvas = None


__all__ = [
    "PotionLab",
    "Essence",
    "EssenceState",
    "Player",
    "Enchanter",
    "Refiner",
    "Cauldron",
    "Bottler",
    "TrashCan",
    "Dispenser",
    "DeliveryWindow",
    "ESSENCE_TYPES",
    "PhysicsConfig",
    "setup_physics_space",
    "add_walls",
    "CollisionHandler",
    "RoundManager",
    "create_default_layout",
    "draw_essence",
    "draw_tool",
    "draw_player",
    "draw_dispenser",
    "draw_ui",
    "render_essence",
]
