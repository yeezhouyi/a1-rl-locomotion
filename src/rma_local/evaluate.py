"""Paired, deterministic deployment evaluation with a fixed scenario suite."""
import argparse
from pathlib import Path
import json
import time

import numpy as np
import torch

from .env import A1Env, OBS_DIM, ROOT
from .models import make_adaptation, Policy, shift_history
from .provenance import provenance
from .settings import settings_dict, task_settings
from .sensors import SensorChannel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["baseline", "oracle", "rma", "zero-latent", "constant-latent"], required=True)
    parser.add_argument("--scenarios", default=str(ROOT / "configs" / "eval_scenarios.json"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--out", default="outputs/eval")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--history-mode", choices=["ordered", "shuffled"], default="ordered")
    args = parser.parse_args()
    if min(args.episodes, args.horizon) <= 0:
        parser.error("episodes and horizon must be positive")
    torch.set_num_threads(1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kind = "baseline" if args.mode == "baseline" else "teacher"
    if "adaptation" not in ckpt and ckpt["config"]["kind"] != kind:
        raise ValueError("Checkpoint kind does not match evaluation mode")
    if args.mode == "baseline" and "adaptation" in ckpt:
        raise ValueError("RMA checkpoint cannot be evaluated as the baseline")
    policy = Policy(kind)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    adaptation = None
    if args.mode == "rma":
        if "adaptation" not in ckpt:
            raise ValueError("RMA mode requires an adaptation checkpoint")
        adaptation = make_adaptation(ckpt["config"].get("architecture", "pooled_cnn"), ckpt["config"].get("history", 50))
        adaptation.load_state_dict(ckpt["adaptation"])
        adaptation.eval()
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    profile = ckpt["config"].get("env_profile", "legacy")
    env = A1Env(randomize=False, horizon=args.horizon, profile=profile)
    # Fixed deployment input, computed from the declared TRAINING distribution.
    # Never sample or inspect actual evaluation dynamics for this ablation.
    constant_latent = None
    if args.mode == "constant-latent":
        ranges = np.asarray(task_settings(profile).train_ranges)
        prior = np.random.default_rng(24680).uniform(ranges[:, 0], ranges[:, 1], (4096, 3))
        prior = (prior - [0.7, 1.0, 1.0]) / [0.3, 1.0, 0.15]
        with torch.inference_mode():
            constant_latent = policy.encoder(torch.as_tensor(prior, dtype=torch.float32)).mean(0, keepdim=True)
    try:
        for scenario in scenarios:
            for episode in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode, options=dict(parameters=scenario["parameters"], command=scenario["command"]))
                sensors = SensorChannel(args.seed + episode + 7654321, scenario.get("sensor_noise_scale", 0.0), scenario.get("sensor_delay_steps", 0))
                obs = sensors.apply(obs)
                history = torch.zeros(1, ckpt["config"].get("history", 50), OBS_DIM)
                history[0, -1] = torch.from_numpy(obs)
                shuffle_rng = np.random.default_rng(args.seed + episode + 123456)
                metrics, latencies, frames = [], [], []
                recovery_streak = 0
                recovery_time = None
                changed = False
                reward_sum = 0.
                fell = False
                for step in range(args.horizon):
                    if "change_step" in scenario and step == scenario["change_step"]:
                        env.set_dynamics(scenario["after_parameters"])
                        changed = True
                    if "push_step" in scenario and scenario["push_step"] <= step < scenario["push_step"] + scenario.get("push_steps", 5):
                        env.data.xfrc_applied[1, :3] = scenario["push_force"]
                    else:
                        env.data.xfrc_applied[:] = 0
                    tensor_obs = torch.from_numpy(obs)[None]
                    start = time.perf_counter()
                    with torch.inference_mode():
                        if args.mode == "oracle":
                            action = policy.deterministic(tensor_obs, torch.from_numpy(env.privileged())[None])
                        elif args.mode == "rma":
                            model_history = history if args.history_mode == "ordered" else history[:, shuffle_rng.permutation(history.shape[1])]
                            action = policy.deterministic(tensor_obs, latent=adaptation(model_history))
                        elif args.mode == "constant-latent":
                            action = policy.deterministic(tensor_obs, latent=constant_latent)
                        elif args.mode == "zero-latent":
                            action = policy.deterministic(tensor_obs, latent=torch.zeros(1, policy.latent_dim))
                        else:
                            # Baseline never reads hidden environment parameters.
                            action = policy.deterministic(tensor_obs)
                    latencies.append((time.perf_counter() - start) * 1000)
                    obs, reward, term, timeout, info = env.step(action[0].numpy())
                    obs = sensors.apply(obs)
                    history = shift_history(history, torch.from_numpy(obs)[None])
                    metrics.append(info)
                    reward_sum += reward
                    if changed:
                        recovery_streak = recovery_streak + 1 if info["velocity_error"] < 0.2 and info["yaw_error"] < 0.3 else 0
                        if recovery_streak >= 25 and recovery_time is None:
                            recovery_time = (step - scenario["change_step"] + 1 - 25) * 0.02
                    if args.video and episode == 0 and step % 2 == 0:
                        frames.append(env.render())
                    if term or timeout:
                        fell = bool(term)
                        break
                if frames:
                    import imageio.v3 as iio
                    iio.imwrite(out / f"{scenario['name']}.mp4", np.stack(frames), fps=25)
                row = dict(scenario=scenario["name"], episode=episode, eval_seed=args.seed + episode, mode=args.mode, steps=len(metrics), fell=fell, completed=not fell and len(metrics) == args.horizon, return_sum=reward_sum, velocity_rmse=float(np.sqrt(np.mean([i["velocity_error"] ** 2 for i in metrics]))), yaw_rmse=float(np.sqrt(np.mean([i["yaw_error"] ** 2 for i in metrics]))), action_delta=float(np.mean([i["action_delta"] for i in metrics])), power=float(np.mean([i["power"] for i in metrics])), forward_velocity=float(np.mean([i["forward_velocity"] for i in metrics])), inference_p95_ms=float(np.percentile(latencies[5:] or latencies, 95)), recovery_s=recovery_time)
                # Survival alone also counts a stationary robot: require tracking.
                row["tracking_success"] = row["completed"] and row["velocity_rmse"] < 0.2 and row["yaw_rmse"] < 0.3
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        env.close()
    summary = {}
    for scenario in scenarios:
        subset = [r for r in rows if r["scenario"] == scenario["name"]]
        summary[scenario["name"]] = {key: {"mean": float(np.mean([r[key] for r in subset])), "std": float(np.std([r[key] for r in subset], ddof=1)) if len(subset) > 1 else None} for key in ["velocity_rmse", "yaw_rmse", "fell", "completed", "tracking_success", "steps", "action_delta", "power", "forward_velocity", "inference_p95_ms"]}
        recovery = [r["recovery_s"] for r in subset if r["recovery_s"] is not None]
        summary[scenario["name"]]["recovery"] = dict(recovered_count=len(recovery), episodes=len(subset), mean_s=float(np.mean(recovery)) if recovery else None)
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(dict(arguments=vars(args), task=settings_dict(profile), provenance=provenance(), scenarios=scenarios, results=summary), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
