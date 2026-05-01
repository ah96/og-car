"""Evaluation metrics for OG-CAR experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from .data_types import Environment, ExplanationFactor
from .ontology import AffordanceOntology
from .planners import PlannerConfig, plan
from .reasoner import apply_intervention, route_overlap_deviation


def precision_at_k(predicted: List[ExplanationFactor], truth: Set[Tuple[str, str]], k: int) -> float:
    """Compute Precision@k."""
    if k <= 0:
        return 0.0
    pred_keys = {f.key for f in predicted[:k]}
    return len(pred_keys & truth) / float(k)


def recall_at_k(predicted: List[ExplanationFactor], truth: Set[Tuple[str, str]], k: int) -> float:
    """Compute Recall@k."""
    if not truth:
        return 0.0
    pred_keys = {f.key for f in predicted[:k]}
    return len(pred_keys & truth) / float(len(truth))


def admissibility_at_k(predicted: List[ExplanationFactor], k: int) -> float:
    """Compute fraction of selected factors that are admissible."""
    if k <= 0:
        return 0.0
    selected = predicted[:k]
    if not selected:
        return 0.0
    return sum(float(f.admissible) for f in selected) / float(k)


def evaluate_selected_interventions(
    env: Environment,
    selected: List[ExplanationFactor],
    planner_config: PlannerConfig,
) -> Dict[str, float]:
    """Evaluate combined top-k interventions.

    Interventions are applied sequentially to a map copy before replanning.
    """
    baseline = plan(env.start, env.goal, env.grid, planner_config)
    baseline_dev = route_overlap_deviation(baseline.path, env.reference_path)

    updated_grid = env.grid.copy()
    objects_by_id = {obj.object_id: obj for obj in env.objects}
    for factor in selected:
        obj = objects_by_id.get(factor.object_id)
        if obj is None:
            continue
        updated_grid = apply_intervention(updated_grid, obj, factor.intervention)

    result = plan(env.start, env.goal, updated_grid, planner_config)
    result_dev = route_overlap_deviation(result.path, env.reference_path)

    feasibility_recovery = float((not baseline.feasible) and result.feasible)
    if baseline.cost < float("inf") and result.cost < float("inf"):
        cost_improvement = max(0.0, baseline.cost - result.cost)
    else:
        cost_improvement = 0.0
    deviation_reduction = max(0.0, baseline_dev - result_dev)
    return {
        "feasibility_recovery": feasibility_recovery,
        "cost_improvement": cost_improvement,
        "deviation_reduction": deviation_reduction,
    }


def minimality_gap(predicted: List[ExplanationFactor], truth: Set[Tuple[str, str]], k: int) -> float:
    """Approximate minimality gap.

    This lightweight version uses one as the oracle size whenever at least one
    ground-truth factor exists. For paper-final experiments, replace this with
    exact subset search on small maps.
    """
    if not truth:
        return 0.0
    selected_count = min(k, len(predicted))
    return float(max(0, selected_count - 1))


def compute_metrics(
    env: Environment,
    predicted: List[ExplanationFactor],
    truth: Set[Tuple[str, str]],
    planner_config: PlannerConfig,
    k: int,
) -> Dict[str, float]:
    """Compute all benchmark metrics for one prediction."""
    intervention_metrics = evaluate_selected_interventions(env, predicted[:k], planner_config)
    return {
        "precision_at_k": precision_at_k(predicted, truth, k),
        "recall_at_k": recall_at_k(predicted, truth, k),
        "admissibility_at_k": admissibility_at_k(predicted, k),
        "minimality_gap": minimality_gap(predicted, truth, k),
        **intervention_metrics,
    }
