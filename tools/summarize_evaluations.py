"""Summarize episode-level evaluation JSONL files."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = [
    "name",
    "scenario",
    "n",
    "velocity_rmse_mean",
    "velocity_rmse_std",
    "yaw_rmse_mean",
    "fall_rate",
    "tracking_success_rate",
    "mean_steps",
    "power_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Evaluation label and directory containing episodes.jsonl; repeatable.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    return parser.parse_args()


def load_evaluation(spec):
    if "=" not in spec:
        raise ValueError(f"--evaluation must be NAME=DIR: {spec}")
    name, directory = spec.split("=", 1)
    if not name or not directory:
        raise ValueError(f"--evaluation must be NAME=DIR: {spec}")
    path = Path(directory) / "episodes.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return name, rows


def mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def summarize(name, rows):
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)
    summaries = []
    for scenario in sorted(by_scenario):
        group = by_scenario[scenario]
        velocities = np.asarray([row["velocity_rmse"] for row in group], dtype=float)
        summaries.append(
            {
                "name": name,
                "scenario": scenario,
                "n": len(group),
                "velocity_rmse_mean": float(np.mean(velocities)),
                "velocity_rmse_std": "" if len(group) == 1 else float(np.std(velocities, ddof=1)),
                "yaw_rmse_mean": mean(group, "yaw_rmse"),
                "fall_rate": mean(group, "fell"),
                "tracking_success_rate": mean(group, "tracking_success"),
                "mean_steps": mean(group, "steps"),
                "power_mean": mean(group, "power"),
            }
        )
    return summaries


def main():
    args = parse_args()
    rows = []
    for spec in args.evaluation:
        name, episodes = load_evaluation(spec)
        rows.extend(summarize(name, episodes))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "comparison.md").open("w", encoding="utf-8") as handle:
        handle.write("# Evaluation comparison\n\n")
        handle.write("These are episode statistics from a single training seed; they do not replace results across training seeds.\n\n")
        handle.write("| " + " | ".join(FIELDS) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(FIELDS)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row[field]) for field in FIELDS) + " |\n")


if __name__ == "__main__":
    main()
