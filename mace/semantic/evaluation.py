"""
mace/semantic/evaluation.py
============================
Evaluation harness for MACE semantic navigation scenarios.

Ground truth definition
-----------------------
For a given (scenario, agent, start, goal) tuple:
  1. Run the planner → get actual_path (may be infeasible).
  2. For every space individual and every ACTIONABLE property:
       Apply the counterfactual value to a clone → re-evaluate.
       A (space, property) pair is ground-truth iff:
         - Restores feasibility (inf → finite cost), OR
         - Reduces actual path cost by > epsilon_cost.
  3. Ground truth = set of (individual_name, property_name) pairs.

Prediction (MACE counterfactuals)
----------------------------------
Uses a unit-weight (topological) graph to enumerate geometrically-short
candidates including currently-blocked routes, then explains each with
CounterfactualEngine.  Returns top-k PropertyChanges ranked by utility.

Baselines
---------
  random       — random shuffle of actionable change candidates
  distance     — rank by graph-edge distance to start node
  semantic     — rank by blocking-affordance severity
  occupancy    — rank by overlap with shortest path edges
  unconstrained — brute-force cost improvement ignoring actionability
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .affordance import AffordanceReasoner
from .counterfactual import CounterfactualEngine, PropertyChange
from .domain import AffordanceType, DoorState
from .navigation import AStarPlanner, NavPath, NavigationGraph
from .ontology import OntologyIndividual


# ---------------------------------------------------------------------------
# Prediction record (analogous to ExplanationFactor in ogcar/data_types.py)
# ---------------------------------------------------------------------------

@dataclass
class SemanticFactor:
    """A ranked explanation factor in the semantic world model."""
    individual_name: str
    property_name:   str
    change:          PropertyChange
    utility:         float
    admissible:      bool   # True if change is actionable (not structural)
    feasible_recovery: bool = False
    cost_improvement:  float = 0.0

    @property
    def key(self) -> Tuple[str, str]:
        return (self.individual_name, self.property_name)


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------

ACTIONABLE_PROPERTIES = {
    "door_state", "is_accessible", "is_hazardous",
    "restricted", "obstacle_density", "crowd_density", "illumination",
}

# Counterfactual values to try for each actionable property
_CF_VALUES: Dict[str, object] = {
    "door_state":       DoorState.OPEN.value,
    "is_accessible":    True,
    "is_hazardous":     False,
    "restricted":       False,
    "obstacle_density": 0.0,
    "crowd_density":    0.2,
    "illumination":     0.8,
}


def compute_ground_truth_semantic(
    graph:         NavigationGraph,
    agent:         OntologyIndividual,
    start:         str,
    goal:          str,
    reasoner:      AffordanceReasoner,
    epsilon_cost:  float = 1.0,
) -> Set[Tuple[str, str]]:
    """
    Compute operational ground-truth (individual, property) pairs.

    A pair is ground-truth iff applying the actionable change to the
    corresponding space individual restores feasibility or improves
    cost by more than epsilon_cost.
    """
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)

    truth: Set[Tuple[str, str]] = set()

    for ind in list(graph.all_space_individuals()):  # materialise — graph mutates during sweep
        for prop, cf_val in _CF_VALUES.items():
            orig_val = ind.get(prop)
            if orig_val is None:
                continue
            if orig_val == cf_val:
                continue

            # Apply change to a clone (never mutate the live individual).
            # clone() appends "_cf" to the name — restore must use clone.name.
            clone = ind.clone()
            clone.set(prop, cf_val)

            _swap_individual(graph, ind.name, clone)
            result = planner.find_path(start, goal, agent)
            _swap_individual(graph, clone.name, ind)  # restore: search by clone's name

            restores = (not baseline.is_feasible) and result.is_feasible
            improves = (
                baseline.is_feasible
                and result.is_feasible
                and (baseline.total_cost - result.total_cost) > epsilon_cost
            )
            if restores or improves:
                truth.add((ind.name, prop))

    return truth


def _swap_individual(
    graph:    NavigationGraph,
    name:     str,
    new_ind:  OntologyIndividual,
) -> None:
    """Replace every edge's space_individual that has the given name in-place."""
    for edge_list in graph.edges.values():
        for edge in edge_list:
            if edge.space_individual.name == name:
                edge.space_individual = new_ind


# ---------------------------------------------------------------------------
# OntofactNavigator → ranked SemanticFactor list
# ---------------------------------------------------------------------------

def predict_ogcar(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    start:    str,
    goal:     str,
    reasoner: AffordanceReasoner,
    k:        int = 3,
) -> List[SemanticFactor]:
    """
    Run the OntofactNavigator counterfactual engine and return top-k factors.

    Uses a unit-weight (topological) graph to enumerate geometrically-short
    candidate alternatives — including currently-blocked routes that cost=inf
    in the affordance graph.  This allows the engine to explain:

    (a) Why a BLOCKED route is infeasible and what minimal changes would
        open it (e.g., door closed, area restricted, corridor too narrow).
    (b) Why a REACHABLE but expensive alternative isn't preferred and what
        changes would reduce its cost (e.g., high crowd, poor illumination).

    Candidates are ranked by counterfactual utility: feasibility-recovery
    scores (inf → finite) beat cost-improvement scores.
    """
    import networkx as nx

    planner   = AStarPlanner(graph, reasoner)
    baseline  = planner.find_path(start, goal, agent)
    cf_engine = CounterfactualEngine(
        reasoner         = reasoner,
        planner          = planner,
        onto_individuals = {ind.name: ind for ind in graph.all_space_individuals()},
    )

    # Build a unit-weight graph: topology only, no affordance cost.
    # This surfaces geometrically-short paths even if currently blocked.
    ug = nx.DiGraph()
    for nid in graph.nodes:
        ug.add_node(nid)
    for fid, elist in graph.edges.items():
        for edge in elist:
            if edge.to_id in graph.nodes:
                ug.add_edge(fid, edge.to_id, weight=1)

    # Collect up to k+2 distinct topological paths
    candidate_seqs: List[List[str]] = []
    try:
        for seq in nx.shortest_simple_paths(ug, start, goal, weight="weight"):
            candidate_seqs.append(seq)
            if len(candidate_seqs) >= k + 2:
                break
    except Exception:
        pass

    if not candidate_seqs:
        return []

    # Skip any sequence identical to the actual chosen path (already explained)
    actual_seq = tuple(baseline.nodes)
    alt_seqs   = [s for s in candidate_seqs if tuple(s) != actual_seq]

    # Evaluate each alternative in the real affordance-weighted world
    alt_paths = [planner.evaluate_sequence(seq, agent) for seq in alt_seqs]

    # explain_why_not(baseline, alt_nodes) answers:
    #   "What minimal changes would make this alternative preferable?"
    cfs: List = []
    seen_seqs: Set[tuple] = {actual_seq}
    for alt_path in alt_paths:
        seq_key = tuple(alt_path.nodes)
        if seq_key in seen_seqs:
            continue
        seen_seqs.add(seq_key)
        cf = cf_engine.explain_why_not(baseline, alt_path.nodes, agent)
        cfs.append(cf)

    # Collect and deduplicate PropertyChanges; assign utility scores
    seen:    Set[Tuple[str, str]] = set()
    factors: List[SemanticFactor] = []

    for cf in cfs:
        for change in cf.changes:
            key = (change.individual_name, change.property_name)
            if key in seen:
                continue
            seen.add(key)
            feasible_recovery = (not baseline.is_feasible) and cf.cf_cost < math.inf
            # Utility: large score for feasibility recovery, otherwise cost delta
            if feasible_recovery:
                utility = 1000.0 + max(0.0, cf.cost_delta if cf.cost_delta < math.inf else 0.0)
            else:
                utility = max(0.0, cf.cost_delta if cf.cost_delta < math.inf else 0.0)
            factors.append(SemanticFactor(
                individual_name  = change.individual_name,
                property_name    = change.property_name,
                change           = change,
                utility          = utility,
                admissible       = change.is_actionable(),
                feasible_recovery= feasible_recovery,
                cost_improvement = max(0.0, cf.cost_delta if cf.cost_delta < math.inf else 0.0),
            ))

    factors.sort(key=lambda f: f.utility, reverse=True)
    return factors[:k]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _all_actionable_candidates(
    graph:    NavigationGraph,
    reasoner: AffordanceReasoner,
    agent:    OntologyIndividual,
) -> List[Tuple[OntologyIndividual, str]]:
    """Return all (individual, property) pairs where applying CF value differs."""
    candidates = []
    for ind in graph.all_space_individuals():
        for prop, cf_val in _CF_VALUES.items():
            orig = ind.get(prop)
            if orig is not None and orig != cf_val:
                candidates.append((ind, prop))
    return candidates


def baseline_random(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    k:        int,
    seed:     int,
) -> List[SemanticFactor]:
    rng = random.Random(seed)
    candidates = _all_actionable_candidates(graph, reasoner, agent)
    rng.shuffle(candidates)
    factors = []
    for i, (ind, prop) in enumerate(candidates[:k]):
        change = PropertyChange(
            individual_name=ind.name, property_name=prop,
            original_value=ind.get(prop),
            counterfactual_value=_CF_VALUES[prop],
        )
        factors.append(SemanticFactor(
            individual_name=ind.name, property_name=prop,
            change=change,
            utility=float(len(candidates) - i),
            admissible=True,
        ))
    return factors


def baseline_distance(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    k:        int,
) -> List[SemanticFactor]:
    """Rank candidates by minimum edge distance from start."""
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, list(graph.nodes.keys())[-1], agent)

    # BFS distance from start (in terms of graph hops)
    from collections import deque
    dist: Dict[str, float] = {start: 0.0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for edge in graph.neighbors(cur):
            if edge.to_id not in dist:
                dist[edge.to_id] = dist[cur] + edge.distance
                queue.append(edge.to_id)

    candidates = _all_actionable_candidates(graph, reasoner, agent)

    def min_dist_to_start(ind: OntologyIndividual) -> float:
        for edge_list in graph.edges.values():
            for edge in edge_list:
                if edge.space_individual.name == ind.name:
                    d = min(
                        dist.get(edge.from_id, math.inf),
                        dist.get(edge.to_id,   math.inf),
                    )
                    return d
        return math.inf

    ranked = sorted(candidates, key=lambda x: min_dist_to_start(x[0]))
    factors = []
    for ind, prop in ranked[:k]:
        change = PropertyChange(
            individual_name=ind.name, property_name=prop,
            original_value=ind.get(prop),
            counterfactual_value=_CF_VALUES[prop],
        )
        factors.append(SemanticFactor(
            individual_name=ind.name, property_name=prop,
            change=change,
            utility=-min_dist_to_start(ind),
            admissible=True,
        ))
    return factors


def baseline_semantic(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    goal:     str,
    k:        int,
) -> List[SemanticFactor]:
    """Rank by affordance-blocking severity on the shortest path."""
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)

    # Collect spaces on the actual path (or all if infeasible)
    if baseline.is_feasible:
        path_spaces = {e.space_individual.name for e in baseline.edges}
    else:
        path_spaces = {
            ind.name for ind in graph.all_space_individuals()
        }

    def score(ind: OntologyIndividual) -> float:
        af = reasoner.compute(ind, agent)
        # Penalise based on which hard-blocking affordances are absent
        s = 0.0
        if AffordanceType.TRAVERSABLE not in af.affordances:
            s += 4.0
        if AffordanceType.PASSABLE not in af.affordances:
            s += 3.0
        if AffordanceType.CLIMBABLE not in af.affordances:
            s += 2.0
        if AffordanceType.OPENABLE not in af.affordances and ind.get("door_state"):
            s += 1.5
        if ind.name in path_spaces:
            s += 2.0
        return s

    candidates = _all_actionable_candidates(graph, reasoner, agent)
    ranked = sorted(candidates, key=lambda x: score(x[0]), reverse=True)
    factors = []
    for ind, prop in ranked[:k]:
        change = PropertyChange(
            individual_name=ind.name, property_name=prop,
            original_value=ind.get(prop),
            counterfactual_value=_CF_VALUES[prop],
        )
        factors.append(SemanticFactor(
            individual_name=ind.name, property_name=prop,
            change=change,
            utility=score(ind),
            admissible=True,
        ))
    return factors


def baseline_occupancy(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    goal:     str,
    k:        int,
) -> List[SemanticFactor]:
    """Rank by overlap between the individual's edges and the shortest path."""
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)
    path_spaces = {e.space_individual.name for e in baseline.edges}

    def overlap(ind: OntologyIndividual) -> int:
        return 1 if ind.name in path_spaces else 0

    candidates = _all_actionable_candidates(graph, reasoner, agent)
    ranked = sorted(candidates, key=lambda x: overlap(x[0]), reverse=True)
    factors = []
    for ind, prop in ranked[:k]:
        change = PropertyChange(
            individual_name=ind.name, property_name=prop,
            original_value=ind.get(prop),
            counterfactual_value=_CF_VALUES[prop],
        )
        factors.append(SemanticFactor(
            individual_name=ind.name, property_name=prop,
            change=change,
            utility=float(overlap(ind)),
            admissible=True,
        ))
    return factors


def baseline_unconstrained(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    goal:     str,
    k:        int,
) -> List[SemanticFactor]:
    """Rank all candidates by raw cost improvement (no actionability filter)."""
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)

    candidates_raw = []
    for ind in list(graph.all_space_individuals()):  # materialise before mutating
        for prop, cf_val in _CF_VALUES.items():
            orig = ind.get(prop)
            if orig is None or orig == cf_val:
                continue

            clone = ind.clone()
            clone.set(prop, cf_val)
            _swap_individual(graph, ind.name, clone)
            result = planner.find_path(start, goal, agent)
            _swap_individual(graph, clone.name, ind)  # restore by clone.name

            restores = (not baseline.is_feasible) and result.is_feasible
            improvement = 1000.0 * float(restores)
            if baseline.is_feasible and result.is_feasible:
                improvement += max(0.0, baseline.total_cost - result.total_cost)

            change = PropertyChange(
                individual_name=ind.name, property_name=prop,
                original_value=orig, counterfactual_value=cf_val,
            )
            candidates_raw.append(SemanticFactor(
                individual_name=ind.name, property_name=prop,
                change=change,
                utility=improvement,
                admissible=prop in ACTIONABLE_PROPERTIES,
                feasible_recovery=restores,
                cost_improvement=max(0.0, (baseline.total_cost - result.total_cost)
                                    if baseline.is_feasible and result.is_feasible else 0.0),
            ))

    candidates_raw.sort(key=lambda f: f.utility, reverse=True)
    return candidates_raw[:k]


def run_baseline(
    method:   str,
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    goal:     str,
    k:        int,
    seed:     int,
) -> List[SemanticFactor]:
    """Dispatch a baseline by name."""
    if method == "random":
        return baseline_random(graph, agent, reasoner, k, seed)
    if method == "distance":
        return baseline_distance(graph, agent, reasoner, start, k)
    if method == "semantic":
        return baseline_semantic(graph, agent, reasoner, start, goal, k)
    if method == "occupancy":
        return baseline_occupancy(graph, agent, reasoner, start, goal, k)
    if method == "unconstrained":
        return baseline_unconstrained(graph, agent, reasoner, start, goal, k)
    raise ValueError(f"Unknown baseline: {method}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(
    predicted: List[SemanticFactor],
    truth:     Set[Tuple[str, str]],
    k:         int,
) -> float:
    if k <= 0:
        return 0.0
    pred_keys = {f.key for f in predicted[:k]}
    return len(pred_keys & truth) / float(k)


def recall_at_k(
    predicted: List[SemanticFactor],
    truth:     Set[Tuple[str, str]],
    k:         int,
) -> float:
    if not truth:
        return 0.0
    pred_keys = {f.key for f in predicted[:k]}
    return len(pred_keys & truth) / float(len(truth))


def admissibility_at_k(predicted: List[SemanticFactor], k: int) -> float:
    if k <= 0 or not predicted:
        return 0.0
    selected = predicted[:k]
    return sum(float(f.admissible) for f in selected) / float(k)


def feasibility_recovery_at_k(predicted: List[SemanticFactor], k: int) -> float:
    """1.0 if at least one top-k factor restores feasibility."""
    return float(any(f.feasible_recovery for f in predicted[:k]))


def cost_improvement_at_k(predicted: List[SemanticFactor], k: int) -> float:
    """Sum of cost improvements from top-k factors."""
    return sum(f.cost_improvement for f in predicted[:k])


def compute_metrics_semantic(
    predicted: List[SemanticFactor],
    truth:     Set[Tuple[str, str]],
    k:         int,
) -> Dict[str, float]:
    return {
        "precision_at_k":       precision_at_k(predicted, truth, k),
        "recall_at_k":          recall_at_k(predicted, truth, k),
        "admissibility_at_k":   admissibility_at_k(predicted, k),
        "feasibility_recovery":  feasibility_recovery_at_k(predicted, k),
        "cost_improvement":      cost_improvement_at_k(predicted, k),
    }
