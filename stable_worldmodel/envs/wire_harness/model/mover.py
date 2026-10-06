"""
MOVER - robot/agent class from the original script
(reduced to the methods used by the Env).
"""

import math
import numpy as np


class Mover:
    """
    Represents a single robot/mover in the simulation.

    What a Mover does:
    - Manages its own position and target
    - Provides distance/angle features for the observation
    - Chooses constraint actions (spacing, collision avoidance)
    - Sets joint velocities in MuJoCo
    """

    def __init__(
        self,
        env,
        mu_index,
        mu_start,
        mu_joint,
        mu_start_move,
        follow,
        max_dist,
        vel,
        cable_connect,
        cable_start_mu,
    ):
        """
        Initializes a Mover.

        Args:
            env: Reference to the Environment
            mu_index: Body ID in MuJoCo (91, 99, 105, 110, 115)
            mu_start: Start position [x, y] in meters
            mu_joint: Joint name prefix (e.g. "slide_joint1")
            mu_start_move: Initial movement direction [x, y]
            follow: True if this mover should follow Mover 0
            max_dist: Maximum allowed distance to others
            vel: Velocity factor
            cable_connect: List of connected cable IDs
            cable_start_mu: Body IDs of the cable start points
        """
        # ========== REWARD TRACKING ==========
        self.reward_total = 0
        self.mean_reward = 0
        self.reward_sum = 0
        self.reward = 0
        self.done = False
        self.reward_list = []

        # ========== COORDINATE TRACKING ==========
        self.coords_x = []
        self.coords_y = []

        # ========== ENVIRONMENT REFERENCE ==========
        self.env = env

        # ========== MOVER PROPERTIES ==========
        self.mu_index = mu_index
        self.mu_start = mu_start
        self.x = mu_start[0]
        self.y = mu_start[1]

        # ========== JOINT CONTROL ==========
        mu_joint_x = mu_joint + 'x'
        self.joint_x = mu_joint_x
        mu_joint_y = mu_joint + 'y'
        self.joint_y = mu_joint_y

        # ========== MOVEMENT PARAMETERS ==========
        self.vel = vel
        self.start_move = mu_start_move
        self.follow = follow
        self.max_dist = max_dist

        # ========== CABLE CONNECTIONS ==========
        self.cable_connect = cable_connect
        self.cable_start_mu = cable_start_mu

        # ========== LOCAL COLLISION MAPS ==========
        # Populated by the Env (WireHarnessEnv._update_local_maps); this is just
        # the storage — it feeds into the observation and grid penalties.
        self.mu_collision_map = np.zeros((5, 5))
        self.mu_cable_collision_map = np.zeros((7, 7))

        # ========== TARGET COORDINATES ==========
        self.x_t = 0
        self.y_t = 0

    def update_pos(self):
        """
        Updates position from MuJoCo data.
        Called every simulation step.
        """
        self.x = self.env.data.xpos[self.mu_index][0]
        self.y = self.env.data.xpos[self.mu_index][1]

    def get_distance(self, x, y, dist_norm=0):
        """
        Computes distance to a point.

        What this computes:
        - Euclidean distance via Pythagoras
        - Optional: normalization to [-1, 1]

        Args:
            x, y: Target point
            dist_norm: Normalization factor (0 = no normalization)

        Returns:
            Distance or normalized distance
        """
        dist = math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

        if dist_norm > 0:
            return (dist - dist_norm / 2) / (dist_norm / 2)
        else:
            return dist

    def get_distance_x(self, x):
        """X distance (signed)"""
        return self.x - x

    def get_distance_y(self, y):
        """Y distance (signed)"""
        return self.y - y

    def get_distance_target(self, norm=True):
        """
        Distance to the target.

        Args:
            norm: True to normalize to [-1, 1]
        """
        dist = math.sqrt((self.x - self.x_t) ** 2 + (self.y - self.y_t) ** 2)

        if norm:
            return (dist - 5 / 2) / (5 / 2)
        else:
            return dist

    def get_angle_target(self, norm=True):
        """Angle to the target (atan2, full angular range [-π, π])"""
        angle = math.atan2((self.y - self.y_t), (self.x - self.x_t))

        if norm:
            return angle / 3.142
        else:
            return angle

    def make_move(self, action):
        """
        Sets the velocity of the joints.

        What happens here:
        - Multiplies the action by the velocity factor
        - Sets joint velocities in MuJoCo

        Args:
            action: [x, y] movement direction (normalized)
        """
        self.env.data.joint(self.joint_x).qvel[0] = self.vel * action[0]
        self.env.data.joint(self.joint_y).qvel[0] = self.vel * action[1]

    def set_target(self, x_t, y_t):
        """Sets new target coordinates"""
        self.x_t = x_t
        self.y_t = y_t

    def choose_constraint_action(self, step, dist):
        """
        Chooses an action based on constraints.

        Priority order:
        1. Follow constraint: maintain distance to Mover 0
        2. Spacing constraint: max distance to others
        3. Collision avoidance: local obstacles

        Returns:
            [x, y] action, or [0, 0] if there's no constraint action
        """
        # ========== CONSTRAINT 1: FOLLOW ==========
        if self.follow and dist > self.max_dist:
            x_dist = self.get_distance_x(self.env.movers[0].x)
            y_dist = self.get_distance_y(self.env.movers[0].y)
            action = [-x_dist / dist, -y_dist / dist]
            return action

        # ========== CONSTRAINT 2: SPACING ==========
        if not self.follow:
            for i in range(self.env.num_agents - 1):
                dist1 = self.get_distance(
                    self.env.movers[i + 1].x, self.env.movers[i + 1].y
                )
                if dist1 > self.env.movers[i + 1].max_dist:
                    x_dist = self.get_distance_x(self.env.movers[i + 1].x)
                    y_dist = self.get_distance_y(self.env.movers[i + 1].y)
                    action = [-x_dist / dist1, -y_dist / dist1]
                    return action

        # ========== CONSTRAINT 3: COLLISION ==========
        if np.sum(self.mu_collision_map) > 1:
            action = self.collision_avoidance()
            return action

        # No constraint action
        return [0, 0]

    def collision_avoidance(self):
        """
        Simple collision avoidance.

        Strategy:
        - Move away from the side with more obstacles
        - Left vs. right and top vs. bottom
        """
        x_dir = (
            0.5
            if np.sum(self.mu_collision_map[:, :1])
            > np.sum(self.mu_collision_map[:, 2:])
            else -0.5
        )
        y_dir = (
            0.5
            if np.sum(self.mu_collision_map[:1, :])
            > np.sum(self.mu_collision_map[2:, :])
            else -0.5
        )

        return [x_dir, y_dir]

    def deterministic_move_t(self):
        """
        Direct movement toward the target.

        What happens here:
        1. Computes the direction vector to the target
        2. Normalizes to a Manhattan distance of 0.5
        """
        x_dist = self.get_distance_x(self.x_t)
        y_dist = self.get_distance_y(self.y_t)

        norm = math.sqrt(x_dist**2 + y_dist**2)
        x_dir = -x_dist / norm
        y_dir = -y_dist / norm

        scaling = 0.5 / (abs(x_dir) + abs(y_dir))
        x_scaled = x_dir * scaling
        y_scaled = y_dir * scaling

        return [x_scaled, y_scaled]
