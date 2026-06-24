"""Plot OG-CAR experiment results using Seaborn."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS = [
    ("precision_at_k", "Precision@k"),
    ("recall_at_k", "Recall@k"),
    ("feasibility_recovery", "Feasibility Recovery@k"),
    ("cost_improvement", "Path-Cost Improvement@k"),
    ("deviation_reduction", "Deviation Reduction@k"),
    ("admissibility_at_k", "Admissibility@k"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot OG-CAR experiment results.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV from run_experiments.py.")
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory for output figures.")
    return parser.parse_args()


def plot_metric(df: pd.DataFrame, metric: str, label: str, output_dir: Path) -> None:
    """Create a bar plot for one metric."""
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="method", y=metric, hue="scenario", errorbar="sd")
    plt.xlabel("Method")
    plt.ylabel(label)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output_path = output_dir / f"{metric}.pdf"
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """Create all result plots."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    for metric, label in METRICS:
        if metric in df.columns:
            plot_metric(df, metric, label, output_dir)

    summary = df.groupby(["scenario", "method"])[[m for m, _ in METRICS if m in df.columns]].agg(["mean", "std"])
    summary.to_csv(output_dir / "summary_statistics.csv")
    print(f"Saved plots and summary statistics to {output_dir}")


if __name__ == "__main__":
    main()
