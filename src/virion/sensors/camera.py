"""Camera abstraction supporting RGB + IR feeds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class CameraSource(Protocol):
    def read(self) -> np.ndarray:
        ...


@dataclass
class CameraDriver:
    source: CameraSource

    def read(self) -> np.ndarray:
        frame = self.source.read()
        return np.asarray(frame)
