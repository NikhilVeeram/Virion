"""Grid-based A*/D* hybrid planner placeholder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import heapq
import numpy as np


@dataclass
class PlannerConfig:
    resolution: float = 0.25
    heuristic_weight: float = 1.2


class GridPlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def _neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = node
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [(x + dx, y + dy) for dx, dy in steps]

    def plan(self, occupancy: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        width, height = occupancy.shape
        open_list: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_list, (0.0, start))
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score = {start: 0.0}

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return self.config.heuristic_weight * (abs(a[0] - b[0]) + abs(a[1] - b[1]))

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                break
            for neighbor in self._neighbors(current):
                x, y = neighbor
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                if occupancy[x, y] > 0.5:
                    continue
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))

        path: List[Tuple[int, int]] = []
        node = goal
        while node in came_from:
            path.append(node)
            node = came_from.get(node)
            if node is None:
                break
        return list(reversed(path))


__all__ = ["GridPlanner", "PlannerConfig"]
