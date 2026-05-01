"""OG-CAR reasoning implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .data_types import Environment, ExplanationFactor, GridObject, Intervention, Path
from .graph import AffordanceRelationGraph, build_graph
from .ontology import AffordanceOntology
from .planners import BLOCKED_COST, PlannerConfig, plan


@dataclass(frozen=True)
class ReasonerConfig:
    """Configuration for OG-CAR scoring."""

    k: int = 2
    lambda_feasibility: float = 1000.0
    lambda_cost: float = 1.0
    lambda_deviation: float = 10.0
    lambda_burden: float = 1.0


def apply_intervention(grid: np.ndarray, obj: GridObject, intervention: Intervention) -> np.ndarray:
    """Return a counterfactual grid after applying an intervention.

    The update rules are deterministic abstractions used for controlled
    experiments. They can later be replaced by probabilistic or learned
    intervention models.
    """
    updated = grid.copy()
    if intervention.affordance in {"movable", "step_aside"}:
        for cell in obj.cells:
            updated[cell] = 1.0
    elif intervention.affordance == "openable":
        for cell in obj.cells:
            updated[cell] = 1.0
    return updated


def route_overlap_deviation(path: Path, reference_path: Optional[Path]) -> float:
    """Compute bounded route-overlap deviation."""
    if not path or not reference_path:
        return 1.0
    path_set = set(path)
    ref_set = set(reference_path)
    union = path_set | ref_set
    if not union:
        return 0.0
    return 1.0 - (len(path_set & ref_set) / len(union))


class OGCarReasoner:
    """Ontology-Guided Counterfactual Affordance Reasoner."""

    def __init__(
        self,
        ontology: AffordanceOntology,
        planner_config: PlannerConfig,
        config: ReasonerConfig | None = None,
    ) -> None:
        self.ontology = ontology
        self.planner_config = planner_config
        self.config = config or ReasonerConfig()

    def explain(self, env: Environment) -> List[ExplanationFactor]:
        """Return ranked OG-CAR explanation factors for an environment."""
        baseline = plan(env.start, env.goal, env.grid, self.planner_config)
        graph = build_graph(env, baseline.path)
        baseline_deviation = route_overlap_deviation(baseline.path, env.reference_path)

        factors: List[ExplanationFactor] = []
        object_by_id = {obj.object_id: obj for obj in env.objects}

        for obj in env.objects:
            for intervention in self.ontology.candidate_interventions(obj):
                if not self.is_admissible(obj, intervention, graph):
                    continue
                factor = self.evaluate_intervention(
                    env=env,
                    obj=obj,
                    intervention=intervention,
                    baseline_cost=baseline.cost,
                    baseline_feasible=baseline.feasible,
                    baseline_deviation=baseline_deviation,
                    graph=graph,
                )
                factors.append(factor)

        factors.sort(key=lambda f: f.utility, reverse=True)
        return factors[: self.config.k]

    def is_admissible(
        self,
        obj: GridObject,
        intervention: Intervention,
        graph: AffordanceRelationGraph,
    ) -> bool:
        """Check semantic, state, and navigation relevance admissibility."""
        if not self.ontology.is_semantically_valid(obj, intervention.affordance):
            return False
        if obj.states.get(intervention.affordance, 1) != 0:
            return False
        relations = graph.relations_for(obj.object_id)
        return bool(relations & {"OnRoute", "BlocksCorridor", "BetweenRobotAndGoal", "NearPath"})

    def evaluate_intervention(
        self,
        env: Environment,
        obj: GridObject,
        intervention: Intervention,
        baseline_cost: float,
        baseline_feasible: bool,
        baseline_deviation: float,
        graph: AffordanceRelationGraph,
    ) -> ExplanationFactor:
        """Evaluate and score one counterfactual intervention."""
        counterfactual_grid = apply_intervention(env.grid, obj, intervention)
        result = plan(env.start, env.goal, counterfactual_grid, self.planner_config)
        counterfactual_deviation = route_overlap_deviation(result.path, env.reference_path)

        feasible_recovery = (not baseline_feasible) and result.feasible
        if baseline_cost == float("inf") and result.cost < float("inf"):
            cost_improvement = 0.0
        else:
            cost_improvement = max(0.0, baseline_cost - result.cost)
        deviation_reduction = max(0.0, baseline_deviation - counterfactual_deviation)
        burden = self.ontology.burden(intervention.affordance)

        utility = (
            self.config.lambda_feasibility * float(feasible_recovery)
            + self.config.lambda_cost * cost_improvement
            + self.config.lambda_deviation * deviation_reduction
            - self.config.lambda_burden * burden
        )

        return ExplanationFactor(
            object_id=obj.object_id,
            object_type=obj.object_type,
            affordance=intervention.affordance,
            intervention=intervention,
            utility=utility,
            admissible=True,
            feasible_recovery=feasible_recovery,
            cost_improvement=cost_improvement,
            deviation_reduction=deviation_reduction,
            burden=burden,
            relations=graph.relations_for(obj.object_id),
        )


def compute_ground_truth(
    env: Environment,
    ontology: AffordanceOntology,
    planner_config: PlannerConfig,
    epsilon_cost: float = 1.0,
    epsilon_deviation: float = 0.01,
) -> Set[Tuple[str, str]]:
    """Compute operational ground-truth object--affordance factors."""
    baseline = plan(env.start, env.goal, env.grid, planner_config)
    baseline_dev = route_overlap_deviation(baseline.path, env.reference_path)
    graph = build_graph(env, baseline.path)
    reasoner = OGCarReasoner(ontology, planner_config)

    truth: Set[Tuple[str, str]] = set()
    for obj in env.objects:
        for intervention in ontology.candidate_interventions(obj):
            # Ground truth is operational, but still restricted to semantic validity.
            if not ontology.is_semantically_valid(obj, intervention.affordance):
                continue
            updated_grid = apply_intervention(env.grid, obj, intervention)
            result = plan(env.start, env.goal, updated_grid, planner_config)
            result_dev = route_overlap_deviation(result.path, env.reference_path)
            restores = (not baseline.feasible) and result.feasible
            improves_cost = baseline.cost < float("inf") and (baseline.cost - result.cost) > epsilon_cost
            improves_dev = (baseline_dev - result_dev) > epsilon_deviation
            if restores or improves_cost or improves_dev:
                truth.add((obj.object_id, intervention.affordance))
    return truth
