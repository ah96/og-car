"""Core data structures for OG-CAR.

The module intentionally keeps the data model simple and explicit so that
experiments remain reproducible and easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]
Path = List[Cell]


class ScenarioType(str, Enum):
    """Supported navigation scenario types."""

    FAILURE = "failure"
    DEVIATION = "deviation"


@dataclass(frozen=True)
class GridObject:
    """Semantic object instance in the local navigation scene.

    Attributes:
        object_id: Stable unique identifier for the object.
        object_type: Semantic type, e.g., 'chair' or 'cabinet'.
        cells: Occupied cells in the grid map.
        affordances: Affordances associated with the object.
        states: Binary affordance state assignment. A value of 0 means the
            affordance is currently unresolved for navigation; 1 means it is
            fulfilled.
    """

    object_id: str
    object_type: str
    cells: Tuple[Cell, ...]
    affordances: Tuple[str, ...]
    states: Dict[str, int]

    @property
    def centroid(self) -> Tuple[float, float]:
        """Return the object centroid in grid coordinates."""
        if not self.cells:
            return (0.0, 0.0)
        r = sum(c[0] for c in self.cells) / len(self.cells)
        c = sum(c[1] for c in self.cells) / len(self.cells)
        return (r, c)


@dataclass(frozen=True)
class Intervention:
    """Counterfactual object--affordance intervention."""

    object_id: str
    affordance: str
    from_state: int = 0
    to_state: int = 1


@dataclass
class ExplanationFactor:
    """Scored explanation factor returned by OG-CAR or a baseline."""

    object_id: str
    object_type: str
    affordance: str
    intervention: Intervention
    utility: float
    admissible: bool
    feasible_recovery: bool = False
    cost_improvement: float = 0.0
    deviation_reduction: float = 0.0
    burden: float = 0.0
    relations: Set[str] = field(default_factory=set)

    @property
    def key(self) -> Tuple[str, str]:
        """Return factor key used for metric comparison."""
        return (self.object_id, self.affordance)


@dataclass
class Environment:
    """Procedural navigation environment."""

    seed: int
    scenario_type: ScenarioType
    grid: "np.ndarray"  # quoted to avoid hard dependency at import time
    start: Cell
    goal: Cell
    objects: List[GridObject]
    direct_corridor: Set[Cell]
    reference_path: Optional[Path] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PlannerResult:
    """Output of a planner call."""

    path: Path
    cost: float

    @property
    def feasible(self) -> bool:
        """Return whether a non-empty finite-cost path exists."""
        return bool(self.path) and self.cost < float("inf")
