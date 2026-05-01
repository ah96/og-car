"""Grid-based planners used by the OG-CAR benchmark."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .data_types import Cell, Path, PlannerResult

BLOCKED_COST = 1e9


@dataclass(frozen=True)
class PlannerConfig:
    """Planner configuration."""

    name: str = "astar"
    heuristic_weight: float = 1.0


def neighbors(cell: Cell, grid: np.ndarray) -> Iterable[Cell]:
    """Yield four-connected valid neighboring cells."""
    row, col = cell
    height, width = grid.shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width:
            if grid[nr, nc] < BLOCKED_COST:
                yield (nr, nc)


def manhattan(a: Cell, b: Cell) -> float:
    """Return Manhattan distance between two cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(parent: Dict[Cell, Optional[Cell]], goal: Cell) -> Path:
    """Reconstruct path from a parent map."""
    path: Path = []
    current: Optional[Cell] = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def path_cost(path: Path, grid: np.ndarray) -> float:
    """Compute path traversal cost."""
    if not path:
        return float("inf")
    return float(sum(grid[cell] for cell in path))


def plan(start: Cell, goal: Cell, grid: np.ndarray, config: PlannerConfig) -> PlannerResult:
    """Plan a path using the configured planner."""
    if config.name == "astar":
        return astar(start, goal, grid, heuristic_weight=1.0)
    if config.name == "weighted_astar":
        return astar(start, goal, grid, heuristic_weight=config.heuristic_weight)
    if config.name == "dijkstra":
        return astar(start, goal, grid, heuristic_weight=0.0)
    if config.name == "bfs":
        return bfs(start, goal, grid)
    raise ValueError(f"Unknown planner: {config.name}")


def astar(start: Cell, goal: Cell, grid: np.ndarray, heuristic_weight: float = 1.0) -> PlannerResult:
    """Run A*/Dijkstra on a weighted grid."""
    if grid[start] >= BLOCKED_COST or grid[goal] >= BLOCKED_COST:
        return PlannerResult(path=[], cost=float("inf"))

    open_heap: List[Tuple[float, Cell]] = []
    heapq.heappush(open_heap, (0.0, start))
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    g_score: Dict[Cell, float] = {start: float(grid[start])}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = reconstruct_path(parent, goal)
            return PlannerResult(path=path, cost=path_cost(path, grid))

        for nxt in neighbors(current, grid):
            tentative = g_score[current] + float(grid[nxt])
            if tentative < g_score.get(nxt, float("inf")):
                parent[nxt] = current
                g_score[nxt] = tentative
                priority = tentative + heuristic_weight * manhattan(nxt, goal)
                heapq.heappush(open_heap, (priority, nxt))

    return PlannerResult(path=[], cost=float("inf"))


def bfs(start: Cell, goal: Cell, grid: np.ndarray) -> PlannerResult:
    """Run breadth-first search on traversable cells."""
    if grid[start] >= BLOCKED_COST or grid[goal] >= BLOCKED_COST:
        return PlannerResult(path=[], cost=float("inf"))

    queue: deque[Cell] = deque([start])
    parent: Dict[Cell, Optional[Cell]] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            path = reconstruct_path(parent, goal)
            return PlannerResult(path=path, cost=float(len(path)))
        for nxt in neighbors(current, grid):
            if nxt not in parent:
                parent[nxt] = current
                queue.append(nxt)
    return PlannerResult(path=[], cost=float("inf"))
