"""Baseline explanation-factor selection methods."""

from __future__ import annotations

import random
from typing import List

from .data_types import Environment, ExplanationFactor, GridObject, Intervention
from .graph import build_graph
from .ontology import AffordanceOntology
from .planners import PlannerConfig, plan
from .reasoner import apply_intervention, route_overlap_deviation


def _factor(obj: GridObject, affordance: str, utility: float, admissible: bool, relations=None) -> ExplanationFactor:
    """Create a baseline explanation factor."""
    intervention = Intervention(object_id=obj.object_id, affordance=affordance)
    return ExplanationFactor(
        object_id=obj.object_id,
        object_type=obj.object_type,
        affordance=affordance,
        intervention=intervention,
        utility=utility,
        admissible=admissible,
        relations=set(relations or set()),
    )


def available_candidates(env: Environment, ontology: AffordanceOntology, semantic_only: bool = True):
    """Yield candidate object--affordance pairs."""
    for obj in env.objects:
        affs = obj.affordances if semantic_only else tuple(set(obj.affordances) | {"movable", "openable", "step_aside"})
        for aff in affs:
            if obj.states.get(aff, 0) == 0:
                yield obj, aff


def random_baseline(env: Environment, ontology: AffordanceOntology, k: int, seed: int) -> List[ExplanationFactor]:
    """Randomly select candidate factors."""
    rng = random.Random(seed)
    candidates = list(available_candidates(env, ontology, semantic_only=True))
    rng.shuffle(candidates)
    return [
        _factor(obj, aff, utility=float(len(candidates) - i), admissible=ontology.is_semantically_valid(obj, aff))
        for i, (obj, aff) in enumerate(candidates[:k])
    ]


def distance_to_path_baseline(env: Environment, ontology: AffordanceOntology, k: int) -> List[ExplanationFactor]:
    """Rank objects by distance to direct route corridor."""
    direct = env.direct_corridor

    def min_dist(obj: GridObject) -> float:
        return min(abs(r - dr) + abs(c - dc) for r, c in obj.cells for dr, dc in direct)

    ranked = sorted(available_candidates(env, ontology, True), key=lambda pair: min_dist(pair[0]))
    return [
        _factor(obj, aff, utility=-min_dist(obj), admissible=ontology.is_semantically_valid(obj, aff))
        for obj, aff in ranked[:k]
    ]


def occupancy_only_baseline(env: Environment, ontology: AffordanceOntology, k: int) -> List[ExplanationFactor]:
    """Rank objects by overlap with direct corridor cells."""
    direct = env.direct_corridor

    def overlap(obj: GridObject) -> int:
        return len(set(obj.cells) & direct)

    ranked = sorted(available_candidates(env, ontology, True), key=lambda pair: overlap(pair[0]), reverse=True)
    return [
        _factor(obj, aff, utility=float(overlap(obj)), admissible=ontology.is_semantically_valid(obj, aff))
        for obj, aff in ranked[:k]
    ]


def semantic_only_baseline(env: Environment, ontology: AffordanceOntology, k: int) -> List[ExplanationFactor]:
    """Rank semantically plausible unresolved objects using graph relevance only."""
    baseline = plan(env.start, env.goal, env.grid, PlannerConfig("astar"))
    graph = build_graph(env, baseline.path)

    def score(obj: GridObject) -> float:
        relations = graph.relations_for(obj.object_id)
        return (
            3.0 * ("OnRoute" in relations)
            + 2.0 * ("BlocksCorridor" in relations)
            + 1.0 * ("NearPath" in relations)
            + 0.5 * ("Near" in relations)
        )

    ranked = sorted(available_candidates(env, ontology, True), key=lambda pair: score(pair[0]), reverse=True)
    return [
        _factor(
            obj,
            aff,
            utility=score(obj),
            admissible=ontology.is_semantically_valid(obj, aff),
            relations=graph.relations_for(obj.object_id),
        )
        for obj, aff in ranked[:k]
    ]


def unconstrained_perturbation_baseline(
    env: Environment,
    ontology: AffordanceOntology,
    planner_config: PlannerConfig,
    k: int,
) -> List[ExplanationFactor]:
    """Rank arbitrary object perturbations without ontology admissibility filtering."""
    baseline = plan(env.start, env.goal, env.grid, planner_config)
    baseline_dev = route_overlap_deviation(baseline.path, env.reference_path)
    candidates = []
    for obj, aff in available_candidates(env, ontology, semantic_only=False):
        intervention = Intervention(obj.object_id, aff)
        updated = apply_intervention(env.grid, obj, intervention)
        result = plan(env.start, env.goal, updated, planner_config)
        dev = route_overlap_deviation(result.path, env.reference_path)
        restores = (not baseline.feasible) and result.feasible
        improvement = 1000.0 * float(restores)
        if baseline.cost < float("inf") and result.cost < float("inf"):
            improvement += max(0.0, baseline.cost - result.cost)
        improvement += 10.0 * max(0.0, baseline_dev - dev)
        candidates.append(_factor(obj, aff, improvement, ontology.is_semantically_valid(obj, aff)))
    candidates.sort(key=lambda f: f.utility, reverse=True)
    return candidates[:k]


def object_attribution_baseline(
    env: Environment,
    ontology: AffordanceOntology,
    planner_config: PlannerConfig,
    k: int,
) -> List[ExplanationFactor]:
    """Estimate object importance by object-level perturbation.

    If an object has multiple affordances, the first listed affordance is used.
    This intentionally lacks affordance-level reasoning.
    """
    baseline = plan(env.start, env.goal, env.grid, planner_config)
    candidates = []
    for obj in env.objects:
        if not obj.affordances:
            continue
        aff = obj.affordances[0]
        intervention = Intervention(obj.object_id, aff)
        updated = apply_intervention(env.grid, obj, intervention)
        result = plan(env.start, env.goal, updated, planner_config)
        restores = (not baseline.feasible) and result.feasible
        improvement = 1000.0 * float(restores)
        if baseline.cost < float("inf") and result.cost < float("inf"):
            improvement += max(0.0, baseline.cost - result.cost)
        candidates.append(_factor(obj, aff, improvement, ontology.is_semantically_valid(obj, aff)))
    candidates.sort(key=lambda f: f.utility, reverse=True)
    return candidates[:k]


def run_baseline(
    method: str,
    env: Environment,
    ontology: AffordanceOntology,
    planner_config: PlannerConfig,
    k: int,
    seed: int,
) -> List[ExplanationFactor]:
    """Dispatch a baseline by name."""
    if method == "random":
        return random_baseline(env, ontology, k, seed)
    if method == "distance":
        return distance_to_path_baseline(env, ontology, k)
    if method == "occupancy":
        return occupancy_only_baseline(env, ontology, k)
    if method == "semantic":
        return semantic_only_baseline(env, ontology, k)
    if method == "unconstrained":
        return unconstrained_perturbation_baseline(env, ontology, planner_config, k)
    if method == "object_attribution":
        return object_attribution_baseline(env, ontology, planner_config, k)
    raise ValueError(f"Unknown baseline method: {method}")
