"""IMU abstraction usable in hardware and simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class IMUSource(Protocol):
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass
class IMUDriver:
    source: IMUSource

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        accel, gyro = self.source.read()
        return np.asarray(accel), np.asarray(gyro)
