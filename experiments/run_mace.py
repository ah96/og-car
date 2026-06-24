"""Run MACE benchmark experiments.

MACE: Minimal-change Affordance Counterfactual Explanations

Each experiment is a (scenario, robot, start, goal) tuple.  For every
combination we run the MACE counterfactual engine + all baselines, compute
metrics against operational ground truth, and save results to CSV.

Example:
    python experiments/run_mace.py --k 3 --output results/mace_results.csv

For a quick smoke test:
    python experiments/run_mace.py --smoke --output results/mace_smoke.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mace.semantic.affordance import AffordanceReasoner
from mace.semantic.evaluation import (
    compute_metrics_semantic,
    compute_ground_truth_semantic,
    predict_ogcar as predict_mace,
    run_baseline,
)
from scenarios.hospital import build_hospital_world
from scenarios.warehouse import build_warehouse_world

SUPPORTED_METHODS = {"mace", "random", "distance", "semantic", "occupancy", "unconstrained"}


def build_tasks(smoke: bool):
    """Return list of (scenario_name, onto, graph, agent_name, agent, start, goal)."""
    tasks = []

    # ── Hospital ───────────────────────────────────────────────────────────────
    onto_h, graph_h, agents_h = build_hospital_world()
    hospital_queries = [
        ("delivery_bot", "entrance", "icu_main"),
        ("cargo_bot",    "entrance", "icu_main"),
        ("legged_bot",   "entrance", "icu_main"),
    ]
    if smoke:
        hospital_queries = hospital_queries[:2]
    for agent_name, start, goal in hospital_queries:
        tasks.append(("hospital", onto_h, graph_h, agent_name, agents_h[agent_name], start, goal))

    # ── Warehouse ──────────────────────────────────────────────────────────────
    onto_w, graph_w, agents_w = build_warehouse_world()
    warehouse_queries = [
        # Feasible baselines
        ("picker_bot",   "loading_bay", "storage_A"),
        ("tracked_bot",  "loading_bay", "mezzanine"),
        ("forklift_bot", "loading_bay", "storage_A"),
        # Infeasible cases (multi-robot differentiation)
        ("picker_bot",   "loading_bay", "mezzanine"),      # ramp too steep for wheeled
        ("forklift_bot", "loading_bay", "narrow_aisle"),   # too wide for narrow aisle
        ("tracked_bot",  "loading_bay", "inspection_zone"),# narrow aisle blocks tracked_bot
    ]
    if smoke:
        warehouse_queries = warehouse_queries[:2]
    for agent_name, start, goal in warehouse_queries:
        tasks.append(("warehouse", onto_w, graph_w, agent_name, agents_w[agent_name], start, goal))

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MACE semantic experiments.")
    parser.add_argument("--k", type=int, default=3, help="Number of explanation factors.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mace", "semantic", "distance", "occupancy", "random", "unconstrained"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/mace_results.csv")
    parser.add_argument("--smoke", action="store_true", help="Run first two tasks per scenario.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = set(args.methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"Unsupported methods: {sorted(unknown)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    reasoner = AffordanceReasoner()
    tasks    = build_tasks(args.smoke)
    rows     = []

    for scenario_name, onto, graph, agent_name, agent, start, goal in tasks:
        print(f"\n[{scenario_name}] {agent_name}: {start} → {goal}")

        truth = compute_ground_truth_semantic(graph, agent, start, goal, reasoner)
        print(f"  ground truth: {len(truth)} factor(s)")

        for method in args.methods:
            if method == "mace":
                predicted = predict_mace(graph, agent, start, goal, reasoner, k=args.k)
            else:
                predicted = run_baseline(
                    method=method,
                    graph=graph,
                    agent=agent,
                    reasoner=reasoner,
                    start=start,
                    goal=goal,
                    k=args.k,
                    seed=args.seed,
                )

            metrics = compute_metrics_semantic(predicted, truth, args.k)
            rows.append({
                "scenario":      scenario_name,
                "agent":         agent_name,
                "start":         start,
                "goal":          goal,
                "method":        method,
                "k":             args.k,
                "num_truth":     len(truth),
                "num_predicted": len(predicted),
                **metrics,
            })
            print(f"  [{method:>14}] P@k={metrics['precision_at_k']:.2f}  "
                  f"R@k={metrics['recall_at_k']:.2f}  "
                  f"Adm={metrics['admissibility_at_k']:.2f}  "
                  f"FeasRec={metrics['feasibility_recovery']:.0f}")

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"\nSaved {len(df)} rows to {output}")
    print(df.groupby("method")[
        ["precision_at_k", "recall_at_k", "feasibility_recovery", "admissibility_at_k"]
    ].mean().to_string())


if __name__ == "__main__":
    main()
