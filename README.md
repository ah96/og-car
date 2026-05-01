# OG-CAR: Ontology-Guided Counterfactual Affordance Reasoning

This repository contains a lightweight, reproducible Python implementation of **Ontology-Guided Counterfactual Affordance Reasoning (OG-CAR)** for actionable robot navigation explanations.

The implementation is designed for the RA-L paper idea:

> A robot explains navigation failures and deviations by identifying object--affordance state changes that would restore feasibility, reduce path cost, or reduce deviation from a reference route.

## Main Features

- Procedural robot-librarian grid-world benchmark.
- Local object and affordance ontology.
- A*, Dijkstra, weighted A*, and BFS planners.
- OG-CAR reasoning with counterfactual replanning.
- Baselines:
  - random
  - distance-to-path
  - occupancy-only
  - semantic-only
  - unconstrained perturbation
  - object-attribution
- Metrics:
  - Precision@k
  - Recall@k
  - Feasibility Recovery@k
  - Path-Cost Improvement@k
  - Deviation Reduction@k
  - Admissibility@k
  - Minimality Gap
- CSV output for experiments.
- Seaborn plotting script.

## Repository Structure

```text
ogcar_affordance_reasoning/
├── README.md
├── requirements.txt
├── run_experiments.py
├── plot_results.py
├── ogcar/
│   ├── __init__.py
│   ├── data_types.py
│   ├── ontology.py
│   ├── environment.py
│   ├── graph.py
│   ├── planners.py
│   ├── reasoner.py
│   ├── baselines.py
│   └── metrics.py
├── results/
└── figures/
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Main Experiments

```bash
python run_experiments.py \
    --num-envs 1000 \
    --scenario both \
    --planner astar \
    --methods ogcar semantic distance occupancy random unconstrained object_attribution \
    --k 2 \
    --seed 42 \
    --output results/main_results.csv
```

For a quick smoke test:

```bash
python run_experiments.py --num-envs 20 --scenario both --planner astar --k 2 --output results/smoke_test.csv
```

## Plot Results

```bash
python plot_results.py \
    --input results/main_results.csv \
    --output-dir figures
```

The script creates Seaborn plots for Precision@k, Recall@k, Feasibility Recovery@k, Cost Improvement@k, and Admissibility@k.

## Notes for Paper Experiments

Recommended experiment groups:

1. **Main comparison**: compare OG-CAR against all baselines.
2. **Semantic clutter**: vary `--extra-distractors` across values such as `0,2,4,6,8,10,12` by running the script multiple times.
3. **Ontology degradation**: use missing/noisy affordance functionality as an extension point in `ontology.py`.
4. **Cross-planner transfer**: generate explanation factors with one planner and evaluate them with another; this is currently prepared structurally but should be expanded in a dedicated script for final experiments.

## Citation Placeholder

If used in the RA-L paper, cite as the implementation of:

```bibtex
@article{halilovic2026ogcar,
  title={Ontology-Guided Counterfactual Affordance Reasoning for Actionable Robot Navigation Explanations},
  author={Halilovic, Amar and Krivic, Senka},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  note={Manuscript in preparation}
}
```

## License

Add your intended license here, e.g., MIT, BSD-3-Clause, or internal research use only.
