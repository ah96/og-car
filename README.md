# MACE: Minimal-change Affordance Counterfactual Explanations

This repository contains the full codebase for the paper:

> **Ontology-Grounded Minimal-Change Counterfactual Explanations for Multi-Robot Navigation**

## What is MACE?

When a robot cannot reach its goal, operators need to know not just *why* it failed but *what they can do about it*. MACE answers this by:

1. Grounding navigation in a formal **OWL 2 ontology** (20 classes, 22 properties) with forward-chaining affordance inference.
2. Running **A\* + Yen's k-shortest paths** on an affordance-weighted semantic navigation graph.
3. Applying **minimal-change counterfactual reasoning** to find the smallest property mutations that restore feasibility or reduce path cost.
4. **Labelling every change** as actionable (open a door, lift a restriction) or structural (widen a corridor, reduce ramp slope), with a three-level effort estimate.

The same pipeline produces robot-specific explanations: affordances — and hence explanations — differ per robot body and capabilities.

## Repository Structure

```text
og-car/
├── mace/                        # Main package
│   ├── semantic/                # Semantic navigation sub-package
│   │   ├── ontology.py          # OWL 2 ontology + SPARQL interface
│   │   ├── domain.py            # Affordance types, enums, build_navigation_ontology()
│   │   ├── affordance.py        # AffordanceReasoner + edge cost function
│   │   ├── navigation.py        # NavigationGraph, NavNode/Edge/Path, AStarPlanner
│   │   ├── counterfactual.py    # CounterfactualEngine, PropertyChange, Counterfactual
│   │   ├── explanation.py       # ExplanationGenerator, NavigationExplanation
│   │   ├── orchestrator.py      # OntofactNavigator (end-to-end interface)
│   │   ├── visualization.py     # draw_navigation_graph() — matplotlib figures
│   │   └── evaluation.py       # Ground truth, baselines, metrics
│   └── __init__.py
├── scenarios/
│   ├── hospital.py              # 9-node hospital (3 robots)
│   └── warehouse.py             # 9-node warehouse (3 robots)
├── experiments/
│   ├── run_mace.py              # Evaluation runner → results/mace_results.csv
│   └── generate_figures.py      # Paper figures → figures/
├── paper/                       # LaTeX paper (gitignored)
├── figures/                     # Generated PDFs/PNGs (gitignored)
├── my_publications/             # Prior work PDFs (gitignored)
├── results/                     # Experiment CSVs
└── requirements.txt
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Experiments

```bash
PYTHONPATH=. python experiments/run_mace.py --k 3 --output results/mace_results.csv
```

## Generate Paper Figures

```bash
PYTHONPATH=. python experiments/generate_figures.py --output-dir figures/ --dpi 250
```

Produces `hospital_<robot>.pdf`, `warehouse_<robot>_<goal>.pdf`, and `narrative_example.txt`.

## Scenarios

| Scenario  | Nodes | Robots | Key constraints |
|-----------|-------|--------|-----------------|
| Hospital  | 9     | `delivery_bot` (0.6 m, arm), `cargo_bot` (1.1 m), `legged_bot` (0.65 m) | Narrow ICU corridor (0.9 m), closed security door, restricted staff corridor |
| Warehouse | 9     | `picker_bot` (max slope 8°), `forklift_bot` (1.4 m wide), `tracked_bot` (max slope 25°) | Steep ramp (18°), narrow aisle (0.85 m), wet inspection zone |

## Baselines

| Method | Description |
|--------|-------------|
| Random | Uniform random ranking of candidate changes |
| Distance | Rank by Euclidean distance from start |
| Occupancy | Rank by overlap with chosen-path segments |
| Semantic | Rank by blocking-affordance severity |
| Unconstrained | Brute-force: re-run A* for every candidate change |
| **MACE** | Minimal-change counterfactual reasoning with actionability filtering |

## Key Results (k=3)

On infeasible tasks with actionable fixes (hospital, cargo/legged bots):

| Method | P@3 | R@3 | FeasRec |
|--------|-----|-----|---------|
| **MACE** | **0.67** | **1.00** | **1.00** |
| Semantic | 0.33 | 0.50 | 0.00 |
| Distance / Occupancy / Random | 0.00 | 0.00 | 0.00 |

MACE is the only method that correctly identifies zero actionable changes on structurally-constrained tasks (Adm@3 = 0.00), and produces no spurious recommendations when no intervention is needed.

## License

Research use only. Contact the author before redistribution.
