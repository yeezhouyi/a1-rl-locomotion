"""Run all three stages sequentially with an explicit finite budget."""
import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ppo-iterations", type=int, default=200)
parser.add_argument("--adapt-iterations", type=int, default=50)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--history", type=int, default=50)
parser.add_argument("--out", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = Path(args.out).resolve()
if out.exists():
    parser.error("Choose a new output directory to prevent overwriting experiments")
if min(args.ppo_iterations, args.adapt_iterations, args.episodes) <= 0 or args.history < 7:
    parser.error("positive budgets and history >= 7 required")
out.mkdir(parents=True)


def run(module, *items):
    command = [sys.executable, "-m", "rma_local." + module, *map(str, items)]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


for kind in ["baseline", "teacher"]:
    run("train", "--kind", kind, "--iterations", args.ppo_iterations, "--seed", args.seed, "--out", out / kind)
run("adapt", "--teacher", out / "teacher/latest.pt", "--iterations", args.adapt_iterations, "--history", args.history, "--seed", args.seed, "--out", out / "adaptation")
for mode, stage in [("baseline", "baseline"), ("oracle", "teacher"), ("rma", "adaptation"), ("zero-latent", "teacher")]:
    run("evaluate", "--checkpoint", out / stage / "latest.pt", "--mode", mode, "--episodes", args.episodes, "--out", out / ("eval_" + mode))
