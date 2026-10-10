"""Fixed terrain and command transitions; distinguish survival, tracking and traversal."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .env import OBS_DIM, ROOT
from .models import Policy, make_adaptation, shift_history
from .terrain import TerrainEnv
from .provenance import provenance
from .settings import task_settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["baseline", "oracle", "rma", "constant"], required=True)
    p.add_argument("--scenarios", default="configs/terrain_confirmation_v5.json")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--seed", type=int, default=71000)
    p.add_argument("--out", required=True)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    if min(args.episodes, args.horizon) <= 0:
        p.error("Positive episode count and horizon required")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kind = "baseline" if args.mode == "baseline" else "teacher"
    if "adaptation" not in ckpt and ckpt["config"]["kind"] != kind:
        raise ValueError("Mode does not match checkpoint")
    policy = Policy(kind); policy.load_state_dict(ckpt["policy"]); policy.eval()
    adapter = None
    if args.mode == "rma":
        adapter = make_adaptation(ckpt["config"].get("architecture", "pooled_cnn"), ckpt["config"].get("history", 50))
        adapter.load_state_dict(ckpt["adaptation"]); adapter.eval()
    profile = ckpt["config"].get("env_profile", "legacy")
    constant = None
    if args.mode == "constant":
        ranges = np.asarray(task_settings(profile).train_ranges)
        prior = np.random.default_rng(24680).uniform(ranges[:,0], ranges[:,1], (4096,3))
        with torch.inference_mode():
            constant = policy.encoder(torch.tensor((prior - [.7,1,1]) / [.3,1,.15], dtype=torch.float32)).mean(0, keepdim=True)
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    env = TerrainEnv(randomize=False, horizon=args.horizon, profile=profile)
    rows, trajectories = [], []
    try:
        for scenario in scenarios:
            schedule = scenario.get("command_schedule", [])
            assert all(0 < c["step"] < args.horizon for c in schedule) or args.horizon < 1000
            assert [c["step"] for c in schedule] == sorted(set(c["step"] for c in schedule))
            for episode in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode, options=dict(terrain=scenario["terrain"], command=scenario["command"], parameters=scenario.get("parameters", [.7,0,1])))
                history = torch.zeros(1, ckpt["config"].get("history", 50), OBS_DIM); history[0,-1] = torch.from_numpy(obs)
                metrics, frames, transitions = [], [], []
                segment = 0
                current_transition, streak = None, 0
                reached_x = 0.
                reward_sum = 0.
                writer = None
                if args.video and episode == 0:
                    import imageio.v2 as iio
                    writer = iio.get_writer(out / f'{scenario["name"]}.mp4', fps=25, macro_block_size=1)
                try:
                    with torch.inference_mode():
                        for step in range(args.horizon):
                            for change in schedule:
                                if step == change["step"]:
                                    env.set_command(change["command"])
                                    obs = env.observe()
                                    history[0,-1,6:9] = torch.from_numpy(env.command)
                                    segment += 1; streak = 0
                                    current_transition = dict(step=step, command=change["command"], response_s=None)
                                    transitions.append(current_transition)
                            ob = torch.from_numpy(obs)[None]
                            if args.mode == "rma":
                                action = policy.deterministic(ob, latent=adapter(history))
                            elif args.mode == "constant":
                                action = policy.deterministic(ob, latent=constant)
                            elif args.mode == "oracle":
                                action = policy.deterministic(ob, torch.from_numpy(env.privileged())[None])
                            else:
                                action = policy.deterministic(ob)
                            obs, reward, term, timeout, info = env.step(action[0].numpy())
                            history = shift_history(history, torch.from_numpy(obs)[None])
                            reward_sum += reward; reached_x = max(reached_x, info["world_x"])
                            info.update(step=step, segment=segment)
                            metrics.append(info)
                            if current_transition:
                                streak = streak+1 if info["velocity_error"] < .2 and info["yaw_error"] < .3 else 0
                                if streak >= 25 and current_transition["response_s"] is None:
                                    current_transition["response_s"] = (step-current_transition["step"]+1-25)*.02
                            if writer and step % 2 == 0:
                                writer.append_data(env.render())
                            if term or timeout:
                                break
                finally:
                    if writer: writer.close()
                completed = not term and len(metrics) == args.horizon
                velocity_rmse = float(np.sqrt(np.mean([m["velocity_error"]**2 for m in metrics])))
                yaw_rmse = float(np.sqrt(np.mean([m["yaw_error"]**2 for m in metrics])))
                tracking = completed and velocity_rmse < .2 and yaw_rmse < .3
                segments = []
                expected_segments = 1 + sum(c["step"] < args.horizon for c in schedule)
                for i in range(expected_segments):
                    selected = [m for m in metrics if m["segment"] == i]
                    segments.append(dict(segment=i, steps=len(selected), velocity_rmse=float(np.sqrt(np.mean([m["velocity_error"]**2 for m in selected]))) if selected else None,
                        yaw_rmse=float(np.sqrt(np.mean([m["yaw_error"]**2 for m in selected]))) if selected else None))
                segment_success = completed and all(s["steps"] > 0 and s["velocity_rmse"] < .2 and s["yaw_rmse"] < .3 for s in segments)
                required_x = scenario.get("required_x")
                corridor_ok = required_x is None or max(abs(m["world_y"]) for m in metrics) < 3.8
                traversed = required_x is None or reached_x >= required_x and corridor_ok
                row = dict(scenario=scenario["name"], episode=episode, eval_seed=args.seed+episode, steps=len(metrics), fell=bool(term), completed=completed,
                    velocity_rmse=velocity_rmse, yaw_rmse=yaw_rmse, tracking_success=tracking, segment_success=segment_success,
                    required_x=required_x, maximum_x=reached_x, final_x=float(env.data.qpos[0]), final_y=float(env.data.qpos[1]), corridor_ok=bool(corridor_ok), traversed=bool(traversed),
                    mission_success=tracking and segment_success and traversed, transitions=transitions, segments=segments, return_sum=reward_sum)
                rows.append(row)
                if episode == 0:
                    trajectories.extend(dict(scenario=scenario["name"], episode=episode, **m) for m in metrics)
                print(json.dumps(row), flush=True)
    finally:
        env.close()
    (out / "episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    (out / "trajectories.jsonl").write_text("".join(json.dumps(r)+"\n" for r in trajectories), encoding="utf-8")
    summary = dict(arguments=vars(args), checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), provenance=provenance(), scenarios=scenarios,
        episodes=len(rows), mission_successes=sum(r["mission_success"] for r in rows), tracking_successes=sum(r["tracking_success"] for r in rows), falls=sum(r["fell"] for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
