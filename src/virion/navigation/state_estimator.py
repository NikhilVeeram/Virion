"""State estimator fusing SLAM + battery awareness."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleState:
    pose: np.ndarray
    velocity: np.ndarray
    battery: float


class StateEstimator:
    def __init__(self) -> None:
        self.state = VehicleState(pose=np.zeros(6), velocity=np.zeros(3), battery=1.0)

    def update_pose(self, pose: np.ndarray) -> None:
        self.state.pose = pose

    def update_battery(self, percentage: float) -> None:
        self.state.battery = percentage

    def get(self) -> VehicleState:
        return self.state
