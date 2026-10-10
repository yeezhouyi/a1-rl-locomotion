"""Finish a bounded local experiment after two already-running PPO jobs.

This helper does not train PPO itself. It waits for the explicitly requested
checkpoints, runs history ablations, evaluates all policies and writes a table.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher", required=True)
parser.add_argument("--baseline", required=True)
parser.add_argument("--iteration", type=int, default=5000)
parser.add_argument("--adapt-iterations", type=int, default=500)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--out", default="outputs/first_comparison")
parser.add_argument("--wait-minutes", type=int, default=30)
args = parser.parse_args()
out = (ROOT / args.out).resolve()
if out.exists():
    parser.error("Choose a fresh output directory")
out.mkdir(parents=True)
status_path = ROOT / "outputs" / "local_run_status.json"
deadline = time.monotonic() + args.wait_minutes * 60


def status(stage, **extra):
    row = dict(stage=stage, output=str(out), **extra)
    status_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row), flush=True)


def wait_checkpoint(filename):
    filename = ROOT / filename
    while time.monotonic() < deadline:
        if filename.exists():
            try:
                checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
                if checkpoint["iteration"] >= args.iteration:
                    return filename
            except (EOFError, RuntimeError, OSError):
                pass  # A checkpoint may still be being written.
        time.sleep(30)
    raise TimeoutError(str(filename))


def launch(label, command):
    handle = (out / (label + "_console.log")).open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, *map(str, command)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    return process, handle


def join(label, job):
    process, handle = job
    code = process.wait()
    handle.close()
    if code:
        raise RuntimeError(f"{label} exited with {code}; see its console log")


def run(label, command):
    join(label, launch(label, command))


try:
    status("waiting_for_teacher")
    teacher = wait_checkpoint(args.teacher)
    status("validating_teacher")
    run("validation_oracle", ["-m", "rma_local.evaluate", "--checkpoint", teacher, "--mode", "oracle", "--scenarios", "configs/validation_scenarios.json", "--seed", "10000", "--episodes", "3", "--out", out / "validation_oracle"])
    status("training_history_50_and_10")
    jobs = []
    for history in [50, 10]:
        label = f"adapt_h{history}"
        command = ["-m", "rma_local.adapt", "--teacher", teacher, "--history", history, "--num-envs", "8", "--iterations", args.adapt_iterations, "--seed", "0", "--out", out / label]
        jobs.append((label, launch(label, command)))
    for label, job in jobs:
        join(label, job)
    evaluations = []
    for label, mode, checkpoint in [("oracle", "oracle", teacher), ("rma_h50", "rma", out / "adapt_h50/latest.pt"), ("rma_h10", "rma", out / "adapt_h10/latest.pt"), ("zero_latent", "zero-latent", teacher)]:
        status("evaluating_" + label)
        target = out / ("eval_" + label)
        run(label, ["-m", "rma_local.evaluate", "--checkpoint", checkpoint, "--mode", mode, "--episodes", args.episodes, "--out", target])
        evaluations.extend(["--evaluation", f"{label}={target}"])
    status("waiting_for_baseline")
    baseline = wait_checkpoint(args.baseline)
    status("evaluating_baseline")
    target = out / "eval_baseline"
    run("baseline", ["-m", "rma_local.evaluate", "--checkpoint", baseline, "--mode", "baseline", "--episodes", args.episodes, "--out", target])
    evaluations.extend(["--evaluation", f"baseline={target}"])
    run("summary", ["tools/summarize_evaluations.py", *evaluations, "--out", out / "comparison"])
    scenarios = json.loads((ROOT / "configs/eval_scenarios.json").read_text(encoding="utf-8"))
    video_scenarios = out / "video_scenarios.json"
    video_scenarios.write_text(json.dumps([r for r in scenarios if r["name"] in ["nominal", "friction_switch"]]), encoding="utf-8")
    status("recording_video")
    run("video", ["-m", "rma_local.evaluate", "--checkpoint", out / "adapt_h50/latest.pt", "--mode", "rma", "--scenarios", video_scenarios, "--episodes", "1", "--video", "--out", out / "videos"])
    for name, directories in dict(baseline=["native_pd_benchmark_s0", "quick_baseline_s0", "baseline_5000_s0"], teacher=["native_pd_teacher_s0", "fast_teacher_s0", "teacher_5000_s0"]).items():
        merged = {}
        for directory in directories:
            path = ROOT / "outputs" / directory / "train.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                merged[row["iteration"]] = row
        (out / (name + "_train.jsonl")).write_text("".join(json.dumps(merged[i]) + "\n" for i in sorted(merged)), encoding="utf-8")
    run("plot", ["tools/plot_training.py", out / "baseline_train.jsonl", out / "teacher_train.jsonl", "--out", out / "training.png"])
    status("complete", comparison=str(out / "comparison/comparison.md"))
except Exception as exc:
    status("failed", error=repr(exc))
    raise
