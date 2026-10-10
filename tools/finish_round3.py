"""Wait for the frozen run, then build predetermined videos and diagnostics."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/third_round"


def main():
    def run(label, command):
        print(label, flush=True)
        with (OUT / (label + "_console.log")).open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)

    def status(stage, **extra):
        (OUT / "delivery_status.json").write_text(json.dumps(dict(stage=stage, **extra), indent=2), encoding="utf-8")

    try:
        status("waiting_for_training")
        while True:
            try:
                state = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(5)
                continue
            if state["stage"] == "confirmation_complete":
                break
            if state["stage"] == "failed":
                raise RuntimeError(state["errors"])
            time.sleep(30)
        status("building_report")
        run("report", ["tools/build_round3_report.py"])
        for seed in range(3):
            run(f"latent_seed_{seed}", ["tools/diagnose_latent.py", "--checkpoint", OUT / f"seed_{seed}/teacher/latest.pt",
                "--out", OUT / f"diagnostics/latent_seed_{seed}.json"])
        run("deployment_benchmark", ["tools/benchmark_deployment.py", "--baseline", OUT / "seed_0/baseline/latest.pt",
            "--cnn", OUT / "seed_0/pooled_cnn/latest.pt", "--ordered", OUT / "seed_0/ordered_mlp/latest.pt",
            "--out", OUT / "diagnostics/deployment_latency.json"])
        # Fixed before viewing any round-three results: ordinary, dynamic and failing stress cases.
        names = ["nominal", "friction_switch", "extrapolation"]
        scenarios = json.loads((ROOT / "configs/confirmation_v3.json").read_text(encoding="utf-8"))
        video_config = OUT / "video_scenarios.json"
        video_config.write_text(json.dumps([s for s in scenarios if s["name"] in names], indent=2), encoding="utf-8")
        methods = [("baseline", "baseline", "baseline"), ("oracle", "oracle", "teacher"),
                   ("cnn", "rma", "pooled_cnn"), ("ordered", "rma", "ordered_mlp")]
        status("recording_fixed_seed_videos")
        for label, mode, folder in methods:
            run("video_" + label, ["-m", "rma_local.evaluate", "--checkpoint", OUT / "seed_0" / folder / "latest.pt",
                "--mode", mode, "--scenarios", video_config, "--episodes", 1, "--seed", 41000, "--video",
                "--out", OUT / ("video_" + label)])
        for name in names:
            clips = []
            for label, _, _ in methods:
                clips.extend(["--clip", f"{label}={OUT / ('video_' + label) / (name + '.mp4')}"])
            run("compose_" + name, ["tools/compare_videos.py", *clips, "--out", OUT / "videos" / (name + ".mp4")])
        # Verify physical evaluation metrics equal the previously recorded episode.
        keys = ["steps", "fell", "completed", "tracking_success", "velocity_rmse", "yaw_rmse", "forward_velocity", "action_delta", "power", "recovery_s"]
        for label, _, _ in methods:
            reference = [json.loads(line) for line in (OUT / "seed_0" / ("eval_"+label) / "episodes.jsonl").read_text().splitlines()]
            video = [json.loads(line) for line in (OUT / ("video_"+label) / "episodes.jsonl").read_text().splitlines()]
            for row in video:
                original = next(r for r in reference if r["scenario"] == row["scenario"] and r["eval_seed"] == row["eval_seed"])
                if any(row[key] != original[key] for key in keys):
                    raise ValueError(f"Video differs from frozen evaluation: {label} {row['scenario']}")
        run("report_with_diagnostics", ["tools/build_round3_report.py"])
        status("ready_for_final_review", videos=names)
        print("ready_for_final_review", flush=True)
    except Exception as exc:
        status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
