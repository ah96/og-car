"""Run OG-CAR benchmark experiments.

Example:
    python run_experiments.py --num-envs 100 --scenario both --planner astar \
        --methods ogcar semantic distance occupancy random unconstrained object_attribution \
        --k 2 --output results/main_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from ogcar.baselines import run_baseline
from ogcar.data_types import ScenarioType
from ogcar.environment import generate_environment
from ogcar.metrics import compute_metrics
from ogcar.ontology import AffordanceOntology
from ogcar.planners import PlannerConfig, plan
from ogcar.reasoner import OGCarReasoner, ReasonerConfig, compute_ground_truth

SUPPORTED_METHODS = {
    "ogcar",
    "random",
    "distance",
    "occupancy",
    "semantic",
    "unconstrained",
    "object_attribution",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run OG-CAR experiments.")
    parser.add_argument("--num-envs", type=int, default=100, help="Number of environments per scenario.")
    parser.add_argument(
        "--scenario",
        choices=["failure", "deviation", "both"],
        default="both",
        help="Scenario type to evaluate.",
    )
    parser.add_argument(
        "--planner",
        choices=["astar", "dijkstra", "weighted_astar", "bfs"],
        default="astar",
        help="Planner used for explanation generation and evaluation.",
    )
    parser.add_argument("--heuristic-weight", type=float, default=2.0, help="Weight for weighted A*.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ogcar", "semantic", "distance", "occupancy", "random"],
        help="Methods to evaluate.",
    )
    parser.add_argument("--k", type=int, default=2, help="Number of explanation factors.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--extra-distractors", type=int, default=0, help="Number of semantic distractors.")
    parser.add_argument("--output", type=str, default="results/main_results.csv", help="CSV output path.")
    return parser.parse_args()


def scenario_list(name: str) -> List[ScenarioType]:
    """Return scenarios to evaluate."""
    if name == "both":
        return [ScenarioType.FAILURE, ScenarioType.DEVIATION]
    return [ScenarioType(name)]


def main() -> None:
    """Run benchmark and save results."""
    args = parse_args()
    unknown = set(args.methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"Unsupported methods: {sorted(unknown)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    ontology = AffordanceOntology.default()
    planner_config = PlannerConfig(name=args.planner, heuristic_weight=args.heuristic_weight)
    reasoner = OGCarReasoner(
        ontology=ontology,
        planner_config=planner_config,
        config=ReasonerConfig(k=args.k),
    )

    rows = []
    for scenario in scenario_list(args.scenario):
        for env_index in range(args.num_envs):
            env_seed = args.seed + env_index + (100000 if scenario == ScenarioType.DEVIATION else 0)
            env = generate_environment(
                seed=env_seed,
                scenario_type=scenario,
                extra_distractors=args.extra_distractors,
            )
            truth = compute_ground_truth(env, ontology, planner_config)
            baseline_result = plan(env.start, env.goal, env.grid, planner_config)

            for method in args.methods:
                if method == "ogcar":
                    predicted = reasoner.explain(env)
                else:
                    predicted = run_baseline(
                        method=method,
                        env=env,
                        ontology=ontology,
                        planner_config=planner_config,
                        k=args.k,
                        seed=env_seed,
                    )
                metrics = compute_metrics(env, predicted, truth, planner_config, args.k)
                rows.append(
                    {
                        "seed": env_seed,
                        "env_index": env_index,
                        "scenario": scenario.value,
                        "planner": args.planner,
                        "method": method,
                        "k": args.k,
                        "extra_distractors": args.extra_distractors,
                        "baseline_feasible": baseline_result.feasible,
                        "num_truth": len(truth),
                        "num_predicted": len(predicted),
                        **metrics,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")
    print(df.groupby("method")[["precision_at_k", "recall_at_k", "feasibility_recovery", "admissibility_at_k"]].mean())


if __name__ == "__main__":
    main()
