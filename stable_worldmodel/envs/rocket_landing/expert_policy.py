"""
Expert Policy wrapper for GNC Controller
Integrates the RocketLandingGNC controller with the stable-worldmodel-flights framework
"""

import time
from collections import deque
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.spatial.transform import Rotation

from stable_worldmodel.policy import BasePolicy


@dataclass
class ControllerParams:
    guidance_hz: float = 5.0
    mpc_horizon: int = 25
    tracking_hz: float = 20.0
    kp_pos: np.ndarray = None
    kd_pos: np.ndarray = None
    attitude_hz: float = 40.0
    kp_att: np.ndarray = None
    ki_att: np.ndarray = None
    kd_att: np.ndarray = None
    thrust_min: float = 0.4
    thrust_max: float = 1.0
    gimbal_limit: float = 0.20
    dry_mass: float = 2.5
    fuel_mass_max: float = 41.0
    g: float = 9.81
    isp: float = 225.0
    max_thrust: float = 427.0
    mpc_dt: float = 0.2
    w_pos_xy: float = 12.0
    w_pos_z: float = 10.0
    w_v_xy: float = 3.0
    w_v_z: float = 6.0
    w_smooth: float = 0.5
    w_acc_mag: float = 0.03
    vz_target_high: float = -1.2
    vz_target_mid: float = -0.8
    vz_target_low: float = -0.2

    def __post_init__(self):
        if self.kp_pos is None:
            self.kp_pos = np.array([1.5, 1.5, 2.5])
        if self.kd_pos is None:
            self.kd_pos = np.array([5.0, 5.0, 6.0])
        if self.kp_att is None:
            self.kp_att = np.array([12.0, 12.0, 6.0])
        if self.ki_att is None:
            self.ki_att = np.array([0.2, 0.2, 0.1])
        if self.kd_att is None:
            self.kd_att = np.array([3.0, 3.0, 2.0])


def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


class FuelEstimator:
    def __init__(self, params: ControllerParams, dt: float):
        self.params = params
        self.dt = dt
        self.fuel_fraction = 1.0
        self.initialized = False

    def update_from_obs(self, fuel_obs):
        if fuel_obs is not None and 0.0 <= fuel_obs <= 1.0:
            self.fuel_fraction = fuel_obs
            self.initialized = True

    def update_from_throttle(self, throttle_norm: float):
        T = clamp01(throttle_norm) * self.params.max_thrust
        mdot = T / (self.params.isp * self.params.g)
        used_mass = mdot * self.dt
        drop = used_mass / self.params.fuel_mass_max
        self.fuel_fraction = max(0.0, self.fuel_fraction - drop)

    def get(self):
        return clamp01(self.fuel_fraction)


class ConvexMPCGuidance:
    def __init__(self, params: ControllerParams):
        self.params = params
        self.N = params.mpc_horizon
        self.dt = params.mpc_dt
        self._last_a = None
        self._solver = None
        self._preferred_solvers = ["ECOS", "SCS"]

    def _vz_target(self, h):
        if h > 15.0:
            return self.params.vz_target_high
        if h > 5.0:
            return self.params.vz_target_mid
        return self.params.vz_target_low

    def compute_reference_trajectory(self, current_state: dict, target_pos: np.ndarray) -> dict:
        t0 = time.perf_counter()
        p0 = current_state["position"].astype(float)
        v0 = current_state["velocity"].astype(float)
        h = float(p0[2])

        fuel_remaining = float(current_state["fuel_remaining"])
        m = self.params.dry_mass + fuel_remaining * self.params.fuel_mass_max
        m = max(m, self.params.dry_mass)

        g = self.params.g
        ez = np.array([0.0, 0.0, 1.0])
        Tmax = self.params.max_thrust / m
        Tmin = (self.params.thrust_min * self.params.max_thrust) / m

        theta_max = self.params.gimbal_limit * (0.9 if h > 5.0 else 0.6)
        tan_th = float(np.tan(theta_max))

        pT = np.array([target_pos[0], target_pos[1], 0.0], dtype=float)
        vT = np.array([0.0, 0.0, self._vz_target(h)], dtype=float)

        p = cp.Variable((3, self.N + 1))
        v = cp.Variable((3, self.N + 1))
        a = cp.Variable((3, self.N))
        u = a + g * ez.reshape(3, 1)

        constraints = []
        constraints += [p[:, 0] == p0, v[:, 0] == v0]

        for k in range(self.N):
            constraints += [p[:, k + 1] == p[:, k] + self.dt * v[:, k]]
            constraints += [v[:, k + 1] == v[:, k] + self.dt * a[:, k]]
            constraints += [cp.norm(u[:, k], 2) <= Tmax]
            constraints += [u[2, k] >= Tmin]
            constraints += [cp.norm(u[0:2, k], 2) <= tan_th * u[2, k]]
        constraints += [p[2, :] >= -0.2]

        cost = 0
        cost += self.params.w_pos_xy * cp.sum_squares(p[0:2, self.N] - pT[0:2])
        cost += self.params.w_pos_z * cp.sum_squares(p[2, self.N] - pT[2])
        cost += self.params.w_v_xy * cp.sum_squares(v[0:2, self.N] - vT[0:2])
        cost += self.params.w_v_z * cp.sum_squares(v[2, self.N] - vT[2])
        cost += self.params.w_acc_mag * cp.sum(cp.sum_squares(u))
        if self.N > 1:
            cost += self.params.w_smooth * cp.sum(cp.sum_squares(a[:, 1:] - a[:, :-1]))

        prob = cp.Problem(cp.Minimize(cost), constraints)

        if self._last_a is not None and self._last_a.shape == (3, self.N):
            a.value = self._last_a
        try_solvers = [s for s in self._preferred_solvers if s in cp.installed_solvers()]
        status = "unstarted"
        try:
            if try_solvers:
                prob.solve(solver=try_solvers[0], warm_start=True, max_iters=500)
            else:
                prob.solve(warm_start=True)
            status = prob.status
        except Exception as e:
            status = f"exception:{type(e).__name__}"

        solve_time = time.perf_counter() - t0

        if status not in ("optimal", "optimal_inaccurate"):
            a0 = np.array([0.0, 0.0, -0.2])
            u0 = a0 + g * ez
            u0_norm = np.linalg.norm(u0) + 1e-9
            thrust_dir = u0 / u0_norm
            throttle_norm = float(np.clip(u0_norm / Tmax, self.params.thrust_min, self.params.thrust_max))
            ref_v = v0 + self.dt * a0
            ref_p = p0 + self.dt * v0
            return {
                "thrust_magnitude": throttle_norm,
                "thrust_direction": thrust_dir,
                "reference_position": ref_p,
                "reference_velocity": ref_v,
                "solve_status": status,
                "solve_time": solve_time,
            }

        a_opt = a.value
        self._last_a = a_opt.copy()

        a0 = a_opt[:, 0]
        u0 = a0 + g * ez
        u0_norm = float(np.linalg.norm(u0) + 1e-9)

        thrust_dir = (u0 / u0_norm).astype(float)
        throttle_norm = float(np.clip(u0_norm / Tmax, self.params.thrust_min, self.params.thrust_max))

        ref_v = v0 + self.dt * a0
        ref_p = p0 + self.dt * v0

        return {
            "thrust_magnitude": throttle_norm,
            "thrust_direction": thrust_dir,
            "reference_position": ref_p,
            "reference_velocity": ref_v,
            "solve_status": status,
            "solve_time": solve_time,
        }


class TrackingController:
    def __init__(self, params: ControllerParams):
        self.params = params

    def compute_control(self, current_state: dict, reference: dict) -> dict:
        pos = current_state["position"]
        vel = current_state["velocity"]
        ref_pos = reference.get("reference_position", np.zeros(3))
        ref_vel = reference.get("reference_velocity", np.zeros(3))
        pos_error = ref_pos - pos
        vel_error = ref_vel - vel
        height = pos[2]

        kp_vec = np.array(self.params.kp_pos, copy=True)
        kd_vec = np.array(self.params.kd_pos, copy=True)
        if height < 20.0:
            lateral_boost = 1.5 if height < 10.0 else 1.25
            damping_boost = 2.0 if height < 10.0 else 1.4
            kp_vec[:2] *= lateral_boost
            kd_vec[:2] *= damping_boost
        if height < 3.0:
            kp_vec[:2] *= 1.6
            kp_scale, kd_scale = 0.8, 1.2
        elif height < 10.0:
            kp_scale, kd_scale = 1.1, 1.1
        else:
            kp_scale, kd_scale = 1.2, 0.9

        accel_cmd = kp_vec * kp_scale * pos_error + kd_vec * kd_scale * vel_error
        if height < 15.0:
            extra_damp = 0.8 if height < 5.0 else 0.5
            accel_cmd[:2] -= extra_damp * vel[:2]
        accel_cmd[2] += self.params.g
        thrust_mag = np.linalg.norm(accel_cmd) + 1e-9
        thrust_dir = accel_cmd / thrust_mag
        current_mass = self.params.dry_mass + current_state["fuel_remaining"] * self.params.fuel_mass_max
        thrust_N = thrust_mag * current_mass
        thrust_normalized = float(
            np.clip(thrust_N / self.params.max_thrust, self.params.thrust_min, self.params.thrust_max)
        )
        return {"thrust_magnitude": thrust_normalized, "thrust_direction": thrust_dir}


class AttitudeController:
    def __init__(self, params: ControllerParams):
        self.params = params
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.dt = 1.0 / params.attitude_hz

    def compute_control(self, current_quat: np.ndarray, desired_thrust_dir: np.ndarray, current_state: dict) -> dict:
        rot = Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
        body_z = rot.apply(np.array([0, 0, 1]))
        cross = np.cross(body_z, desired_thrust_dir)
        s = np.linalg.norm(cross)
        angle_error = np.arcsin(np.clip(s, -1.0, 1.0))
        if s > 1e-6:
            axis = cross / s
            error_body = rot.inv().apply(axis * angle_error)
        else:
            error_body = np.zeros(3)
        height = current_state["position"][2]
        max_integral = 0.3 if height < 5.0 else 0.5
        self.integral_error += error_body * self.dt
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        derivative_error = (error_body - self.prev_error) / self.dt
        self.prev_error = error_body.copy()
        if height < 3.0:
            kp_scale, ki_scale, kd_scale = 1.3, 1.5, 1.2
        elif height < 10.0:
            kp_scale, ki_scale, kd_scale = 1.0, 1.0, 1.0
        else:
            kp_scale, ki_scale, kd_scale = 0.9, 0.8, 0.9
        control = (
            self.params.kp_att * kp_scale * error_body
            + self.params.ki_att * ki_scale * self.integral_error
            + self.params.kd_att * kd_scale * derivative_error
        )
        gimbal_x = -float(np.clip(control[0], -self.params.gimbal_limit, self.params.gimbal_limit))
        gimbal_y = -float(np.clip(control[1], -self.params.gimbal_limit, self.params.gimbal_limit))
        return {"gimbal_x": gimbal_x, "gimbal_y": gimbal_y, "angle_error": angle_error}

    def reset(self):
        self.integral_error[:] = 0.0
        self.prev_error[:] = 0.0


class RocketLandingGNC:
    def __init__(self, params: ControllerParams = None, angle_representation: str = "quaternion"):
        if params is None:
            params = ControllerParams()
        self.params = params
        self.angle_representation = angle_representation
        self.guidance = ConvexMPCGuidance(params)
        self.tracking = TrackingController(params)
        self.attitude = AttitudeController(params)
        self.step_count = 0
        self.guidance_interval = int(params.attitude_hz / params.guidance_hz)
        self.tracking_interval = int(params.attitude_hz / params.tracking_hz)
        self.guidance_cmd = None
        self.tracking_cmd = None
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.telemetry = {
            "guidance_solve_times": deque(maxlen=100),
            "control_errors": deque(maxlen=1000),
        }
        self.dt_att = 1.0 / self.params.attitude_hz
        self.fuel_estimator = FuelEstimator(self.params, self.dt_att)

    def reset(self):
        self.step_count = 0
        self.guidance_cmd = None
        self.tracking_cmd = None
        self.attitude.reset()
        self.fuel_estimator = FuelEstimator(self.params, self.dt_att)

    def compute_control(self, obs_dict: dict) -> np.ndarray:
        pos = obs_dict["position"]
        vel = obs_dict["velocity"]
        quat = obs_dict["quaternion"]
        ang_vel = obs_dict["angular_velocity"]
        fuel_obs = obs_dict.get("fuel_obs")
        target_rel = obs_dict["target_rel"]

        if fuel_obs is not None:
            self.fuel_estimator.update_from_obs(fuel_obs)

        self.target_position = pos + target_rel
        fuel_for_control = self.fuel_estimator.get()

        current_state = {
            "position": pos,
            "velocity": vel,
            "quaternion": quat,
            "angular_velocity": ang_vel,
            "fuel_remaining": fuel_for_control,
        }

        if self.step_count % self.guidance_interval == 0:
            self.guidance_cmd = self.guidance.compute_reference_trajectory(current_state, self.target_position)
            if "solve_time" in self.guidance_cmd:
                self.telemetry["guidance_solve_times"].append(self.guidance_cmd["solve_time"])

        if self.step_count % self.tracking_interval == 0:
            if self.guidance_cmd is not None:
                self.tracking_cmd = self.tracking.compute_control(current_state, self.guidance_cmd)
            else:
                self.tracking_cmd = {"thrust_magnitude": 0.5, "thrust_direction": np.array([0, 0, 1.0])}

        if self.tracking_cmd is not None:
            desired_thrust_dir = self.tracking_cmd["thrust_direction"]
            thrust_magnitude = float(self.tracking_cmd["thrust_magnitude"])
            attitude_cmd = self.attitude.compute_control(quat, desired_thrust_dir, current_state)
            self.telemetry["control_errors"].append(attitude_cmd["angle_error"])
        else:
            attitude_cmd = {"gimbal_x": 0.0, "gimbal_y": 0.0}
            thrust_magnitude = 0.0

        height = pos[2]
        damping = 0.8 if height < 5.0 else 0.5
        finlet_x = -float(np.clip(vel[0] * damping, -1.0, 1.0))
        finlet_y = -float(np.clip(vel[1] * damping, -1.0, 1.0))
        finlet_roll = 0.0

        has_fuel = fuel_for_control > 0.01
        ignition = 1.0 if (thrust_magnitude > self.params.thrust_min and has_fuel) else 0.0

        throttle = float(np.clip(thrust_magnitude, 0.0, 1.0))
        gimbal_x = float(attitude_cmd["gimbal_x"])
        gimbal_y = float(attitude_cmd["gimbal_y"])

        action = np.array([finlet_x, finlet_y, finlet_roll, ignition, throttle, gimbal_x, gimbal_y], dtype=np.float32)
        self.step_count += 1
        self._last_throttle = throttle
        return action

    def post_step_update(self):
        if hasattr(self, "_last_throttle"):
            self.fuel_estimator.update_from_throttle(self._last_throttle)

    def get_fuel_report(self):
        return self.fuel_estimator.get()

    def get_telemetry(self) -> dict:
        return {
            "avg_guidance_solve_time": float(np.mean(self.telemetry["guidance_solve_times"]))
            if self.telemetry["guidance_solve_times"]
            else 0.0,
            "max_guidance_solve_time": float(np.max(self.telemetry["guidance_solve_times"]))
            if self.telemetry["guidance_solve_times"]
            else 0.0,
            "avg_angle_error": float(np.mean(self.telemetry["control_errors"]))
            if self.telemetry["control_errors"]
            else 0.0,
        }


def parse_observation(observation: np.ndarray, angle_rep: str = "quaternion", pad_position=None):
    """Parse PyFlyt rocket observation into structured fields.

    Args:
        observation: The observation vector from the environment
        angle_rep: Angle representation ("quaternion" or "euler")
        pad_position: Position of the landing pad (if moving), otherwise assumes [0, 0, 0]
    """

    obs = np.asarray(observation, dtype=float).flatten()
    if obs.size < 30:
        padded = np.zeros(30, dtype=float)
        padded[: obs.size] = obs
        obs = padded

    ang_vel = obs[0:3]

    if angle_rep == "quaternion":
        raw_quat_xyzw = obs[3:7]
        rot = Rotation.from_quat(raw_quat_xyzw)
        quat = np.array([raw_quat_xyzw[3], raw_quat_xyzw[0], raw_quat_xyzw[1], raw_quat_xyzw[2]], dtype=float)
        idx = 7
    else:
        raw_eul = obs[3:6]
        rot = Rotation.from_euler("xyz", raw_eul)
        q = rot.as_quat()
        quat = np.array([q[3], q[0], q[1], q[2]], dtype=float)
        idx = 6

    vel_body = obs[idx : idx + 3]
    vel = rot.apply(vel_body)
    pos = obs[idx + 3 : idx + 6]
    aux = obs[idx + 13 : idx + 22]  # skip previous action (7 values)

    fuel_obs = float(np.clip(aux[5], 0.0, 1.0))

    # Compute relative target position
    if pad_position is not None:
        target_rel = pad_position - pos
    else:
        target_rel = -pos  # Assume pad at origin

    return {
        "position": pos,
        "velocity": vel,
        "quaternion": quat,
        "angular_velocity": ang_vel,
        "fuel_obs": fuel_obs,
        "target_rel": target_rel,
    }


class ExpertPolicy(BasePolicy):
    def __init__(self, controller_params=None, **kwargs):
        super().__init__(**kwargs)

        # Create the GNC controller with provided or default parameters
        self._controller_params = controller_params or ControllerParams()
        self.controllers = [self._make_controller()]  # TODO num_envs!

    def _make_controller(self):
        return RocketLandingGNC(params=self._controller_params, angle_representation="quaternion")

    def _ensure_controller_count(self, count: int) -> None:
        if count <= 0:
            raise ValueError("Controller count must be positive.")

        if len(self.controllers) < count:
            for _ in range(count - len(self.controllers)):
                self.controllers.append(self._make_controller())
        elif len(self.controllers) > count:
            self.controllers = self.controllers[:count]

    def get_action(self, info_dict, **kwargs):
        """
        Returns:
            action: numpy array of shape (7,) containing:
                [finlet_x, finlet_y, finlet_roll, ignition, throttle, gimbal_x, gimbal_y]
        """
        obs = info_dict["state"] if "state" in info_dict else info_dict["observation"]
        batched = obs.ndim > 1
        obs_batch = obs if batched else obs[None, :]

        # Extract pad position if available (for moving pad support)
        pad_position = info_dict.get("pad_position", None)


        self._ensure_controller_count(obs_batch.shape[0])

        actions = []
        for idx, single_obs in enumerate(obs_batch):
            # Extract pad position for this specific environment
            pad_pos_for_env = pad_position[idx] if pad_position is not None else None
            state_dict = parse_observation(single_obs, "quaternion", pad_position=pad_pos_for_env)
            controller = self.controllers[idx]
            action = controller.compute_control(state_dict)
            controller.post_step_update()
            actions.append(action)

        actions = np.stack(actions, axis=0)
        return actions if batched else actions[0]

    def reset(self):
        for controller in self.controllers:
            controller.reset()

    def get_telemetry(self):
        if len(self.controllers) == 1:
            return self.controllers[0].get_telemetry()
        return [controller.get_telemetry() for controller in self.controllers]

    def get_fuel_report(self):
        if len(self.controllers) == 1:
            return self.controllers[0].get_fuel_report()
        return [controller.get_fuel_report() for controller in self.controllers]
