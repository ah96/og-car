"""Affordance-relation graph construction for OG-CAR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

from .data_types import Cell, Environment, GridObject, Path


@dataclass
class AffordanceRelationGraph:
    """Lightweight local affordance-relation graph."""

    object_relations: Dict[str, Set[str]] = field(default_factory=dict)

    def relations_for(self, object_id: str) -> Set[str]:
        """Return relations for an object."""
        return set(self.object_relations.get(object_id, set()))


def _is_near_path(obj: GridObject, path: Path, radius: int = 1) -> bool:
    """Return whether an object is near a path."""
    path_set = set(path)
    for row, col in obj.cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if (row + dr, col + dc) in path_set:
                    return True
    return False


def _direction_relation(start: Cell, obj: GridObject) -> str:
    """Compute coarse object direction relative to robot start pose."""
    row, col = obj.centroid
    sr, sc = start
    delta_row = row - sr
    delta_col = col - sc
    if abs(delta_col) >= abs(delta_row):
        return "InFrontOf" if delta_col >= 0 else "Behind"
    return "RightOf" if delta_row > 0 else "LeftOf"


def build_graph(env: Environment, baseline_path: Path | None = None) -> AffordanceRelationGraph:
    """Construct a local affordance-relation graph for an environment."""
    direct = env.direct_corridor
    baseline_set = set(baseline_path or [])
    relations: Dict[str, Set[str]] = {}

    for obj in env.objects:
        obj_rel: Set[str] = set()
        obj_rel.add(_direction_relation(env.start, obj))

        object_cells = set(obj.cells)
        if object_cells & direct:
            obj_rel.add("OnRoute")
            obj_rel.add("BetweenRobotAndGoal")
        if baseline_set and _is_near_path(obj, list(baseline_set), radius=1):
            obj_rel.add("NearPath")
        if object_cells & direct:
            obj_rel.add("BlocksCorridor")
        if _distance_to_start(env.start, obj) <= 5.0:
            obj_rel.add("Near")
        relations[obj.object_id] = obj_rel

    return AffordanceRelationGraph(object_relations=relations)


def _distance_to_start(start: Cell, obj: GridObject) -> float:
    """Return Manhattan distance from start to object centroid."""
    row, col = obj.centroid
    return abs(row - start[0]) + abs(col - start[1])
