"""Frozen terrain curriculum and dynamic-command control experiment."""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="outputs/terrain_round")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=False)
    seeds = [0] if args.smoke else [0,1,2]
    plan = dict(seeds=seeds, updates=2 if args.smoke else 2000, adapt_updates=2 if args.smoke else 300,
        episodes=1 if args.smoke else 3, horizon=30 if args.smoke else 1000, eval_seed=71000,
        methods=["old_baseline", "old_teacher", "old_ordered", "commands_teacher", "terrain_teacher", "terrain_ordered", "terrain_constant"],
        learning_rate=1e-4, task_suites=["commands_v5", "terrain_v5"], smoke=args.smoke,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    plan["new_control_steps"] = len(seeds)*(2*plan["updates"]+plan["adapt_updates"])*256
    plan["eval_episodes"] = len(seeds)*7*12*plan["episodes"]
    (out/"execution_plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    snapshot = out/"source_snapshot"
    paths = list((ROOT/"src").rglob("*.py")) + list((ROOT/"configs").glob("*.json")) + list((ROOT/"tools").glob("*.py"))
    paths += [ROOT/"docs/terrain_plan.md", ROOT/"docs/opensource_locomotion.md", ROOT/"THIRD_PARTY_NOTICES.md", ROOT/"licenses/legged_gym.txt", ROOT/"assets/unitree_a1/manifest.json", ROOT/"requirements-lock.txt"]
    hashes = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        dest = snapshot/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest); hashes[relative.as_posix()] = sha(source)
    (out/"source_snapshot.json").write_text(json.dumps(dict(path=str(snapshot),hashes=hashes),indent=2),encoding="utf-8")
    state = dict(stage="running",seeds={},plan=plan); lock=threading.Lock()

    def status(seed,stage,**extra):
        with lock:
            state["seeds"][str(seed)] = dict(stage=stage,**extra)
            temp=out/"status.tmp"; temp.write_text(json.dumps(state,indent=2),encoding="utf-8"); temp.replace(out/"status.json")
        print(f"seed={seed} stage={stage}",flush=True)

    def worker(seed):
        folder=out/f"seed_{seed}"; folder.mkdir()
        parent=ROOT/f"outputs/third_round/seed_{seed}"
        models=dict(old_baseline=parent/"baseline/latest.pt",old_teacher=parent/"teacher/latest.pt",old_ordered=parent/"ordered_mlp/latest.pt")
        (folder/"parents.json").write_text(json.dumps({k:dict(path=str(v),sha256=sha(v)) for k,v in models.items()},indent=2),encoding="utf-8")

        def run(label,command):
            status(seed,label)
            full=[sys.executable,*map(str,command)]
            (folder/f"{label}_command.json").write_text(json.dumps(full,indent=2),encoding="utf-8")
            with (folder/f"{label}_console.log").open("w",encoding="utf-8") as stream:
                subprocess.run(full,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        try:
            for task_suite in plan["task_suites"]:
                label="commands_teacher" if task_suite=="commands_v5" else "terrain_teacher"
                run("train_"+label,["-m","rma_local.train","--kind","teacher","--resume",models["old_teacher"],"--seed",seed,
                    "--iterations",plan["updates"],"--env-profile","robust_v2","--learning-rate",plan["learning_rate"],
                    "--task-suite",task_suite,"--curriculum-offset",0,"--save-every",plan["updates"],"--out",folder/label])
                models[label]=folder/label/"latest.pt"
            run("adapt_terrain",["-m","rma_local.adapt","--teacher",models["terrain_teacher"],"--seed",seed,
                "--iterations",plan["adapt_updates"],"--architecture","ordered_mlp","--history",50,"--out",folder/"terrain_ordered"])
            models["terrain_ordered"]=folder/"terrain_ordered/latest.pt"; models["terrain_constant"]=models["terrain_teacher"]
            for label in plan["methods"]:
                mode="baseline" if label=="old_baseline" else "rma" if "ordered" in label else "constant" if label=="terrain_constant" else "oracle"
                run("eval_"+label,["-m","rma_local.evaluate_terrain","--checkpoint",models[label],"--mode",mode,
                    "--scenarios","configs/terrain_confirmation_v5.json","--episodes",plan["episodes"],"--horizon",plan["horizon"],
                    "--seed",plan["eval_seed"],"--out",folder/("eval_"+label)])
            status(seed,"complete")
        except Exception as exc:
            status(seed,"failed",error=repr(exc)); raise
    errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(worker,s):s for s in seeds}
        for future in as_completed(futures):
            try: future.result()
            except Exception as exc: errors.append(dict(seed=futures[future],error=repr(exc)))
    state.update(stage="failed" if errors else "experiment_complete",errors=errors)
    (out/"status.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
    if errors: raise RuntimeError(errors)
    print("experiment_complete",flush=True)


if __name__=="__main__":
    main()
