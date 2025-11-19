"""PID control utilities for the thruster stabilization loop."""
from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class PIDConfig:
    kp: float
    ki: float
    kd: float
    setpoint: float = 0.0
    integral_limit: float = 1.0


class PIDController:
    def __init__(self, config: PIDConfig) -> None:
        self.config = config
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.perf_counter()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.perf_counter()

    def update(self, measurement: float) -> float:
        now = time.perf_counter()
        dt = max(now - self.prev_time, 1e-4)
        error = self.config.setpoint - measurement
        self.integral += error * dt
        self.integral = max(min(self.integral, self.config.integral_limit), -self.config.integral_limit)
        derivative = (error - self.prev_error) / dt
        output = self.config.kp * error + self.config.ki * self.integral + self.config.kd * derivative
        self.prev_error = error
        self.prev_time = now
        return output


__all__ = ["PIDController", "PIDConfig"]
