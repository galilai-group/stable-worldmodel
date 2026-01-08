from typing import Protocol

import numpy as np


class Quantizable(Protocol):
    """Protocol for quantizable action spaces."""

    def quantize(self, action: np.ndarray) -> np.ndarray: ...

    def dequantize(self, quantized_action: np.ndarray) -> np.ndarray: ...

    @property
    def action_shape(self) -> tuple[int, ...]: ...


class PolarQuantizer(Quantizable):
    def __init__(self, num_bins_radial: int, num_bins_angular: int, max_action_distance: float):
        self.num_bins_radial = num_bins_radial
        self.num_bins_angular = num_bins_angular
        self.max_action_distance = max_action_distance

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self.num_bins_radial, self.num_bins_angular)

    def quantize(self, action: np.ndarray) -> np.ndarray:
        dx, dy = action
        radius = np.linalg.norm(action)
        theta = np.arctan2(dy, dx) % (2 * np.pi)

        radius_bin = int(
            np.clip(radius / self.max_action_distance * self.num_bins_radial, 0, self.num_bins_radial - 1)
        )
        theta_bin = int(np.clip(theta / (2 * np.pi) * self.num_bins_angular, 0, self.num_bins_angular - 1))
        return np.array([radius_bin, theta_bin], dtype=np.int32)

    def dequantize(self, quantized_action: np.ndarray) -> np.ndarray:
        radius_bin, theta_bin = quantized_action

        new_radius = (radius_bin + 0.5) / self.num_bins_radial * self.max_action_distance
        theta = (theta_bin + 0.5) / self.num_bins_angular * 2 * np.pi

        dx = new_radius * np.cos(theta)
        dy = new_radius * np.sin(theta)

        return np.array([dx, dy], dtype=np.float32)
