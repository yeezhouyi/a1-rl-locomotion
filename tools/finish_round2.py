"""After development acceptance, train H50 and run the frozen confirmation set."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher", default="outputs/second_round/teacher_v2/model_007000.pt")
parser.add_argument("--baseline", default="outputs/second_round/baseline_v2/model_007000.pt")
parser.add_argument("--adapt-iterations", type=int, default=750)
parser.add_argument("--episodes", type=int, default=5)
args = parser.parse_args()
out = ROOT / "outputs/second_round"


def status(stage, **extra):
    (out / "status.json").write_text(json.dumps(dict(stage=stage, **extra), indent=2), encoding="utf-8")
    print(stage, flush=True)


def run(label, command):
    with (out / (label + "_console.log")).open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)


try:
    if not (out / "development_review.json").exists():
        raise RuntimeError("Development review must be recorded first")
    review = json.loads((out / "development_review.json").read_text(encoding="utf-8"))
    if not review.get("accept_teacher_for_adaptation"):
        raise RuntimeError("Teacher not accepted for adaptation")
    if (out / "adapt_h50").exists():
        raise FileExistsError("Adaptation directory exists; avoid overwriting evidence")
    status("training_adaptation_h50")
    run("adapt_h50", ["-m", "rma_local.adapt", "--teacher", args.teacher, "--iterations", args.adapt_iterations, "--history", "50", "--out", out / "adapt_h50"])
    status("validating_adaptation")
    run("dev_rma_v2", ["-m", "rma_local.evaluate", "--checkpoint", out / "adapt_h50/latest.pt", "--mode", "rma", "--scenarios", "configs/validation_v2.json", "--seed", "11000", "--episodes", "3", "--out", out / "dev_rma_v2"])
    evaluations = []
    models = [("baseline_original", "baseline", "outputs/baseline_5000_s0/model_005000.pt"), ("teacher_original", "oracle", "outputs/teacher_5000_s0/model_005000.pt"), ("rma_original", "rma", "outputs/first_comparison/adapt_h50/latest.pt"), ("baseline_v2", "baseline", args.baseline), ("teacher_v2", "oracle", args.teacher), ("rma_v2", "rma", out / "adapt_h50/latest.pt"), ("zero_v2", "zero-latent", args.teacher)]
    for label, mode, checkpoint in models:
        status("confirming_" + label)
        target = out / ("confirm_" + label)
        if target.exists():
            raise FileExistsError(target)
        run("confirm_" + label, ["-m", "rma_local.evaluate", "--checkpoint", checkpoint, "--mode", mode, "--scenarios", "configs/confirmation_v2.json", "--seed", "31000", "--episodes", args.episodes, "--out", target])
        evaluations += ["--evaluation", f"{label}={target}"]
    run("confirmation_summary", ["tools/summarize_evaluations.py", *evaluations, "--out", out / "confirmation_comparison"])
    scenarios = json.loads((ROOT / "configs/confirmation_v2.json").read_text(encoding="utf-8"))
    selected = [s for s in scenarios if s["name"] in ["confirm_nominal", "confirm_weak_motor"]]
    video_config = out / "video_scenarios.json"
    video_config.write_text(json.dumps(selected), encoding="utf-8")
    for label, mode, checkpoint in models:
        if label not in ["baseline_v2", "teacher_v2", "rma_v2"]:
            continue
        status("recording_" + label)
        run("video_" + label, ["-m", "rma_local.evaluate", "--checkpoint", checkpoint, "--mode", mode, "--scenarios", video_config, "--seed", "31000", "--episodes", "1", "--video", "--out", out / ("video_" + label)])
    for scenario in ["confirm_nominal", "confirm_weak_motor"]:
        run("compose_" + scenario, ["tools/compare_videos.py", "--clip", f"baseline_v2={out / 'video_baseline_v2' / (scenario + '.mp4')}", "--clip", f"oracle_v2={out / 'video_teacher_v2' / (scenario + '.mp4')}", "--clip", f"rma_v2={out / 'video_rma_v2' / (scenario + '.mp4')}", "--out", out / "videos" / (scenario + ".mp4")])
    status("confirmation_complete")
except Exception as exc:
    status("failed", error=repr(exc))
    raise
