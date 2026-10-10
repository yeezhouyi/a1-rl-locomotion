"""Frozen-endpoint three-seed experiment; --smoke checks the same full chain."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = [("baseline", "baseline", "baseline"), ("oracle", "oracle", "teacher"),
           ("cnn", "rma", "pooled_cnn"), ("ordered", "rma", "ordered_mlp"),
           ("constant", "constant-latent", "teacher"), ("zero", "zero-latent", "teacher"),
           ("ordered_shuffled", "rma", "ordered_mlp")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/third_round")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    seeds = [0] if args.smoke else [0, 1, 2]
    baseline_updates, fork_update, teacher_updates, adapt_updates = (3, 2, 2, 2) if args.smoke else (7000, 5000, 2000, 750)
    plan = dict(seeds=seeds, baseline_updates=baseline_updates, fork_update=fork_update,
                teacher_updates=teacher_updates, adapt_updates=adapt_updates,
                eval_episodes=1 if args.smoke else 5, horizon=30 if args.smoke else 1000,
                env_profile="robust_v2", methods=METHODS, eval_seed=41000,
                teacher_curriculum_offset=fork_update, teacher_learning_rate=1e-4,
                smoke=args.smoke, started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "execution_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    snapshot = (ROOT / "outputs/source_snapshots/round3_training"
                if not args.smoke and args.out == "outputs/third_round"
                else out / "source_snapshot")
    snapshot.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").glob("*.json")) + list((ROOT / "tools").glob("*.py"))
    sources += [ROOT / "docs/round3_plan.md", ROOT / "assets/unitree_a1/manifest.json", ROOT / "pyproject.toml"]
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    (snapshot / "snapshot_manifest.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (out / "source_snapshot.json").write_text(json.dumps(dict(path=str(snapshot), hashes=hashes), indent=2), encoding="utf-8")
    lock = threading.Lock()
    state = dict(stage="running", seeds={}, plan=plan)

    def status(seed, stage, **extra):
        with lock:
            state["seeds"][str(seed)] = dict(stage=stage, **extra)
            target = out / "status.json"
            temp = out / "status.tmp"
            temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temp.replace(target)
        print(f"seed={seed} stage={stage}", flush=True)

    def worker(seed):
        folder = out / f"seed_{seed}"
        folder.mkdir()

        def run(label, command):
            status(seed, label)
            full = [sys.executable, *map(str, command)]
            (folder / (label + "_command.json")).write_text(json.dumps(full, indent=2), encoding="utf-8")
            with (folder / (label + "_console.log")).open("w", encoding="utf-8") as stream:
                subprocess.run(full, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)

        try:
            run("train_baseline", ["-m", "rma_local.train", "--kind", "baseline", "--seed", seed,
                "--iterations", baseline_updates, "--save-every", 1 if args.smoke else 500,
                "--env-profile", "robust_v2", "--out", folder / "baseline"])
            warm = folder / "teacher_initial.pt"
            run("warmstart", ["tools/warmstart_teacher.py", "--baseline", folder / f"baseline/model_{fork_update:06d}.pt", "--out", warm])
            run("train_teacher", ["-m", "rma_local.train", "--kind", "teacher", "--seed", seed,
                "--resume", warm, "--iterations", teacher_updates, "--save-every", 1 if args.smoke else 500,
                "--env-profile", "robust_v2", "--curriculum-offset", fork_update,
                "--learning-rate", "0.0001", "--out", folder / "teacher"])
            for architecture in ["pooled_cnn", "ordered_mlp"]:
                run("adapt_" + architecture, ["-m", "rma_local.adapt", "--teacher", folder / "teacher/latest.pt",
                    "--seed", seed, "--history", 50, "--architecture", architecture,
                    "--iterations", adapt_updates, "--out", folder / architecture])
            for label, mode, checkpoint_folder in METHODS:
                command = ["-m", "rma_local.evaluate", "--checkpoint", folder / checkpoint_folder / "latest.pt",
                    "--mode", mode, "--scenarios", "configs/confirmation_v3.json", "--seed", 41000,
                    "--episodes", plan["eval_episodes"], "--horizon", plan["horizon"], "--out", folder / ("eval_" + label)]
                if label == "ordered_shuffled":
                    command += ["--history-mode", "shuffled"]
                run("eval_" + label, command)
            status(seed, "complete")
        except Exception as exc:
            status(seed, "failed", error=repr(exc))
            raise

    errors = []
    with ThreadPoolExecutor(max_workers=len(seeds)) as executor:
        futures = {executor.submit(worker, seed): seed for seed in seeds}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(dict(seed=futures[future], error=repr(exc)))
    state["stage"] = "failed" if errors else "confirmation_complete"
    state["errors"] = errors
    (out / "status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(errors)
    print(state["stage"], flush=True)


if __name__ == "__main__":
    main()
