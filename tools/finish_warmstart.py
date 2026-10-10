"""Validate a baseline-initialized teacher, then finish only after review."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/second_round"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=["validate", "finish"], required=True)
args = parser.parse_args()
teacher = OUT / "teacher_from_baseline/model_007000.pt"


def status(stage, **extra):
    (OUT / "warmstart_status.json").write_text(json.dumps(dict(stage=stage, **extra), indent=2), encoding="utf-8")
    print(stage, flush=True)


def run(label, command):
    with (OUT / (label + "_console.log")).open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)


def evaluate(label, mode, checkpoint, confirm=False, video=False):
    target = OUT / label
    if target.exists():
        raise FileExistsError(target)
    command = ["-m", "rma_local.evaluate", "--checkpoint", checkpoint, "--mode", mode, "--scenarios", "configs/confirmation_v2.json" if confirm else "configs/validation_v2.json", "--seed", "31000" if confirm else "11000", "--episodes", "5" if confirm else "3", "--out", target]
    if video:
        command += ["--video"]
    run(label, command)


try:
    if args.phase == "validate":
        status("waiting_for_teacher")
        deadline = time.monotonic() + 20 * 60
        while time.monotonic() < deadline:
            if teacher.exists():
                try:
                    ckpt = torch.load(teacher, map_location="cpu", weights_only=False)
                    if ckpt["iteration"] == 7000:
                        break
                except (EOFError, OSError, RuntimeError):
                    pass
            time.sleep(10)
        else:
            raise TimeoutError(teacher)
        status("validating_teacher")
        evaluate("dev_teacher_warmstart", "oracle", teacher)
        status("awaiting_development_review")
    else:
        review = json.loads((OUT / "warmstart_review.json").read_text(encoding="utf-8"))
        if not review["accept_teacher_for_adaptation"]:
            raise RuntimeError("Warmstart teacher not accepted")
        adapt = OUT / "adapt_h50_warmstart"
        if adapt.exists():
            raise FileExistsError(adapt)
        status("training_adaptation")
        run("adapt_h50_warmstart", ["-m", "rma_local.adapt", "--teacher", teacher, "--iterations", "750", "--history", "50", "--out", adapt])
        status("validating_adaptation")
        evaluate("dev_rma_warmstart", "rma", adapt / "latest.pt")
        for label, mode, checkpoint in [("teacher_warmstart", "oracle", teacher), ("rma_warmstart", "rma", adapt / "latest.pt"), ("zero_warmstart", "zero-latent", teacher)]:
            status("confirming_" + label)
            evaluate("confirm_" + label, mode, checkpoint, confirm=True)
        for label, mode, checkpoint in [("teacher_warmstart", "oracle", teacher), ("rma_warmstart", "rma", adapt / "latest.pt")]:
            status("recording_" + label)
            run("video_" + label, ["-m", "rma_local.evaluate", "--checkpoint", checkpoint, "--mode", mode, "--scenarios", OUT / "video_scenarios.json", "--seed", "31000", "--episodes", "1", "--video", "--out", OUT / ("video_" + label)])
        for scenario in ["confirm_nominal", "confirm_weak_motor"]:
            run("compose_warmstart_" + scenario, ["tools/compare_videos.py", "--clip", f"baseline_v2={OUT / 'video_baseline_v2' / (scenario + '.mp4')}", "--clip", f"oracle_warmstart={OUT / 'video_teacher_warmstart' / (scenario + '.mp4')}", "--clip", f"rma_warmstart={OUT / 'video_rma_warmstart' / (scenario + '.mp4')}", "--out", OUT / "videos" / ("warmstart_" + scenario + ".mp4")])
        status("complete")
except Exception as exc:
    status("failed", error=repr(exc))
    raise
