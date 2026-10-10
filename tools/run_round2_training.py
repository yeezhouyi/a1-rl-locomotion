"""Bounded matched fine-tuning; leaves model acceptance to development review."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/second_round"
OUT.mkdir(parents=True, exist_ok=False)


def status(stage, **details):
    (OUT / "status.json").write_text(json.dumps(dict(stage=stage, **details), indent=2), encoding="utf-8")
    print(stage, flush=True)


def launch(label, args):
    stream = (OUT / f"{label}_console.log").open("w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, *map(str, args)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    return p, stream


def join(label, job):
    p, stream = job
    result = p.wait()
    stream.close()
    if result:
        raise RuntimeError(f"{label} failed ({result}); inspect its console log")


try:
    status("training_matched_v2")
    jobs = []
    for kind in ["teacher", "baseline"]:
        jobs.append((kind, launch(kind, ["-m", "rma_local.train", "--kind", kind, "--resume", f"outputs/{kind}_5000_s0/model_005000.pt", "--env-profile", "robust_v2", "--learning-rate", "0.0001", "--iterations", "2000", "--save-every", "500", "--out", OUT / f"{kind}_v2"])))
    for label, job in jobs:
        join(label, job)
    for label, mode, checkpoint in [("teacher_original", "oracle", "outputs/teacher_5000_s0/model_005000.pt"), ("baseline_original", "baseline", "outputs/baseline_5000_s0/model_005000.pt"), ("teacher_v2", "oracle", OUT / "teacher_v2/model_007000.pt"), ("baseline_v2", "baseline", OUT / "baseline_v2/model_007000.pt")]:
        status("validating_" + label)
        join(label, launch("dev_" + label, ["-m", "rma_local.evaluate", "--checkpoint", checkpoint, "--mode", mode, "--scenarios", "configs/validation_v2.json", "--seed", "11000", "--episodes", "3", "--out", OUT / f"dev_{label}"]))
    status("awaiting_development_review")
except Exception as exc:
    status("failed", error=repr(exc))
    raise
