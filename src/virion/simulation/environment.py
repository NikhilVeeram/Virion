"""Hooks into ROS 2 / Gazebo simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class SimulationBridge:
    publish: Callable[[str, Dict], None]
    subscribe: Callable[[str, Callable[[Dict], None]], None]

    def send_control(self, topic: str, command: Dict) -> None:
        self.publish(topic, command)

    def register_sensor(self, topic: str, callback: Callable[[Dict], None]) -> None:
        self.subscribe(topic, callback)
