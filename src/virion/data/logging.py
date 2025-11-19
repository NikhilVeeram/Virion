"""Data logging helpers for dataset capture."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json


@dataclass
class DataLogger:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def log_sensor(self, name: str, payload: Dict[str, Any]) -> None:
        path = self.root / f"{name}.jsonl"
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")
