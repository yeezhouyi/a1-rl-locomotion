"""Fixed paired entropy intervention and seeded sensor stress benchmark."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/fourth_round")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    seeds = [0] if args.smoke else [0, 1, 2]
    plan = dict(seeds=seeds, updates=2 if args.smoke else 750, num_envs=8, steps=32,
                episodes=1 if args.smoke else 3, horizon=20 if args.smoke else 1000,
                eval_seed=61000, learning_rate=1e-4, entropy_coefficient=0.005,
                methods=["baseline", "oracle", "ordered", "constant", "gaussian_teacher", "squashed_teacher"],
                smoke=args.smoke, started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    plan["new_ppo_steps"] = len(seeds) * 2 * plan["updates"] * 256
    plan["eval_episodes"] = len(seeds) * 6 * 8 * plan["episodes"]
    (out / "execution_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    snapshot = out / "source_snapshot"
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").glob("*.json")) + list((ROOT / "tools").glob("*.py"))
    paths += [ROOT / "docs/round4_plan.md", ROOT / "assets/unitree_a1/manifest.json", ROOT / "requirements-lock.txt"]
    hashes = {}
    for path in paths:
        dest = snapshot / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    (out / "source_snapshot.json").write_text(json.dumps(dict(path=str(snapshot), hashes=hashes), indent=2), encoding="utf-8")
    lock = threading.Lock()
    state = dict(stage="running", seeds={}, plan=plan)

    def status(seed, stage, **extra):
        with lock:
            state["seeds"][str(seed)] = dict(stage=stage, **extra)
            temp = out / "status.tmp"
            temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temp.replace(out / "status.json")
        print(f"seed={seed} stage={stage}", flush=True)

    def worker(seed):
        folder = out / f"seed_{seed}"
        folder.mkdir()
        parent = ROOT / f"outputs/third_round/seed_{seed}"
        models = dict(baseline=parent / "baseline/latest.pt", oracle=parent / "teacher/latest.pt",
                      ordered=parent / "ordered_mlp/latest.pt", constant=parent / "teacher/latest.pt")
        (folder / "parent_checkpoints.json").write_text(json.dumps({k: dict(path=str(v), sha256=digest(v)) for k, v in models.items()}, indent=2), encoding="utf-8")

        def run(label, command):
            status(seed, label)
            full = [sys.executable, *map(str, command)]
            (folder / f"{label}_command.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
            with (folder / f"{label}_console.log").open("w", encoding="utf-8") as log:
                subprocess.run(full, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)

        def evaluate(label, mode, model):
            run("eval_" + label, ["-m", "rma_local.evaluate", "--checkpoint", model, "--mode", mode,
                "--scenarios", "configs/stress_v4.json", "--seed", 61000, "--episodes", plan["episodes"],
                "--horizon", plan["horizon"], "--out", folder / ("eval_" + label)])

        try:
            for label, mode in [("baseline", "baseline"), ("oracle", "oracle"), ("ordered", "rma"), ("constant", "constant-latent")]:
                evaluate(label, mode, models[label])
            for mode in ["gaussian", "squashed"]:
                target = folder / f"{mode}_teacher"
                run("train_" + mode, ["-m", "rma_local.train", "--kind", "teacher", "--resume", models["oracle"],
                    "--seed", seed, "--iterations", plan["updates"], "--learning-rate", plan["learning_rate"],
                    "--env-profile", "robust_v2", "--curriculum-offset", 7000, "--entropy-mode", mode,
                    "--entropy-coefficient", plan["entropy_coefficient"], "--save-every", plan["updates"], "--out", target])
                evaluate(mode + "_teacher", "oracle", target / "latest.pt")
            status(seed, "complete")
        except Exception as exc:
            status(seed, "failed", error=repr(exc))
            raise

    errors = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(worker, seed): seed for seed in seeds}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(dict(seed=futures[future], error=repr(exc)))
    state.update(stage="failed" if errors else "experiment_complete", errors=errors)
    (out / "status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(errors)
    print("experiment_complete", flush=True)


if __name__ == "__main__":
    main()
