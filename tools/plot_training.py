from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("logs", nargs="+")
parser.add_argument("--out", default="outputs/training.png")
args = parser.parse_args()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for filename in args.logs:
    rows = [json.loads(line) for line in Path(filename).read_text().splitlines()]
    for ax, key in zip(axes, ["reward", "velocity_error"]):
        if key in rows[0]:
            ax.plot([r.get("total_steps", r["iteration"]) for r in rows], [r[key] for r in rows], label=Path(filename).stem if Path(filename).name != "train.jsonl" else Path(filename).parent.name)
            ax.set_ylabel(key)
            ax.set_xlabel("Environment steps (PPO) / iteration (adaptation)")
            ax.grid(alpha=0.3)
            ax.legend()
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=160)
