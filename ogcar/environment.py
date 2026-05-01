"""Procedural robot-librarian environment generator."""

from __future__ import annotations

import random
from typing import List, Set, Tuple

import numpy as np

from .data_types import Cell, Environment, GridObject, ScenarioType
from .planners import BLOCKED_COST

OBJECT_TYPES = ("chair", "door", "cabinet", "cart", "person", "table")


def create_base_grid(height: int = 15, width: int = 25) -> tuple[np.ndarray, Cell, Cell, Set[Cell], Set[Cell]]:
    """Create a simple map with a direct corridor and a detour corridor."""
    grid = np.full((height, width), BLOCKED_COST, dtype=float)
    start = (height // 2, 1)
    goal = (height // 2, width - 2)

    direct: Set[Cell] = {(height // 2, col) for col in range(1, width - 1)}
    upper: Set[Cell] = set()
    for col in range(1, width - 1):
        upper.add((height // 2 - 4, col))
    for row in range(height // 2 - 4, height // 2 + 1):
        upper.add((row, 1))
        upper.add((row, width - 2))

    for cell in direct | upper:
        grid[cell] = 1.0
    return grid, start, goal, direct, upper


def object_affordances(object_type: str) -> Tuple[str, ...]:
    """Return default affordances for an object type."""
    mapping = {
        "chair": ("movable",),
        "door": ("openable",),
        "cabinet": ("openable", "movable"),
        "cart": ("movable",),
        "person": ("step_aside",),
        "table": ("movable",),
        "shelf": tuple(),
    }
    return mapping.get(object_type, tuple())


def make_object(index: int, object_type: str, cells: Tuple[Cell, ...]) -> GridObject:
    """Create a grid object with all affordances initially unresolved."""
    affordances = object_affordances(object_type)
    states = {aff: 0 for aff in affordances}
    return GridObject(
        object_id=f"{object_type}_{index}",
        object_type=object_type,
        cells=cells,
        affordances=affordances,
        states=states,
    )


def place_object(grid: np.ndarray, obj: GridObject, object_type: str) -> None:
    """Write an unresolved object's cost into the grid."""
    if object_type == "door":
        cost = 5.0
    elif object_type == "person":
        cost = 8.0
    else:
        cost = BLOCKED_COST
    for cell in obj.cells:
        grid[cell] = cost


def generate_environment(
    seed: int,
    scenario_type: ScenarioType,
    extra_distractors: int = 0,
    height: int = 15,
    width: int = 25,
) -> Environment:
    """Generate a deterministic procedural robot-librarian environment."""
    rng = random.Random(seed)
    grid, start, goal, direct, upper = create_base_grid(height, width)
    objects: List[GridObject] = []

    # Place route-relevant shortcut objects on the direct corridor.
    if scenario_type == ScenarioType.FAILURE:
        relevant_positions = [(height // 2, width // 2 - 1), (height // 2, width // 2)]
    else:
        # Deviation: direct route is obstructed but upper detour remains available.
        relevant_positions = [(height // 2, width // 2)]

    relevant_types = ["chair", "cabinet"] if scenario_type == ScenarioType.FAILURE else ["cabinet"]
    for idx, (obj_type, pos) in enumerate(zip(relevant_types, relevant_positions)):
        obj = make_object(idx, obj_type, (pos,))
        objects.append(obj)
        place_object(grid, obj, obj_type)

    # In failure cases, also block the detour to make feasibility recovery meaningful.
    if scenario_type == ScenarioType.FAILURE:
        detour_block = (height // 2 - 4, width // 2)
        obj = make_object(len(objects), "cart", (detour_block,))
        objects.append(obj)
        place_object(grid, obj, "cart")

    # Add distractors to free cells not on the central route-critical positions.
    free_cells = [tuple(cell) for cell in zip(*np.where(grid < BLOCKED_COST))]
    rng.shuffle(free_cells)
    used = {cell for obj in objects for cell in obj.cells}
    for _ in range(extra_distractors):
        candidates = [cell for cell in free_cells if cell not in used and cell not in {start, goal}]
        if not candidates:
            break
        cell = rng.choice(candidates)
        used.add(cell)
        obj_type = rng.choice(OBJECT_TYPES)
        obj = make_object(len(objects), obj_type, (cell,))
        objects.append(obj)
        # Distractors are intentionally placed but do not always become hard blockers.
        # This simulates semantic clutter without necessarily changing the route.
        if rng.random() < 0.4:
            place_object(grid, obj, obj_type)

    reference_path = sorted(direct, key=lambda c: c[1])
    return Environment(
        seed=seed,
        scenario_type=scenario_type,
        grid=grid,
        start=start,
        goal=goal,
        objects=objects,
        direct_corridor=direct,
        reference_path=reference_path,
        metadata={"extra_distractors": extra_distractors},
    )
