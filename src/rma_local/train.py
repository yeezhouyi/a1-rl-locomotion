"""Small, inspectable PPO baseline and privileged RMA base-policy trainer."""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import time

import numpy as np
import torch

from .env import BatchEnv
from .models import Policy
from .ppo import gae, squashed_entropy
from .provenance import provenance
from .settings import settings_dict, task_settings


@dataclass
class TrainConfig:
    kind: str = "baseline"
    seed: int = 0
    num_envs: int = 8
    steps: int = 32
    iterations: int = 1000
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 3e-4
    device: str = "cpu"
    out: str = "outputs/baseline_s0"
    resume: str = ""
    save_every: int = 50
    threads: int = 1
    env_profile: str = "legacy"
    curriculum_offset: int = 0
    entropy_mode: str = "gaussian"
    entropy_coefficient: float = 0.005
    task_suite: str = "flat"


def tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def save_checkpoint(path, policy, optimizer, config, iteration, total_steps):
    torch.save(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(), config=asdict(config), iteration=iteration, total_steps=total_steps), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in asdict(TrainConfig()).items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = TrainConfig(**vars(parser.parse_args()))
    settings = task_settings(args.env_profile)
    if args.kind not in ("baseline", "teacher") or min(args.num_envs, args.steps, args.iterations, args.save_every) <= 0:
        parser.error("kind must be baseline/teacher, and counts must be positive")
    if args.entropy_mode not in ("gaussian", "squashed") or args.entropy_coefficient < 0:
        parser.error("entropy mode must be gaussian/squashed and coefficient nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    entropy_rng = torch.Generator(device=device).manual_seed(args.seed + 987654)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    policy = Policy(args.kind).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    start_iteration, total_steps = 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["config"]["kind"] != args.kind:
            raise ValueError("Cannot resume a different policy kind")
        policy.load_state_dict(checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Adam's saved parameter group would otherwise silently override CLI LR.
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        start_iteration = checkpoint["iteration"]
        total_steps = checkpoint["total_steps"]
    metadata = dict(config=asdict(args), task=settings_dict(args.env_profile), provenance=provenance(), resumed=bool(args.resume), parent_profile=checkpoint["config"].get("env_profile", "legacy") if args.resume else None, note="Resume restores weights/optimizer, not simulator or random-number state. Curriculum uses explicit offset; CLI learning rate is applied.")
    (out / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    initial_curriculum = min(args.curriculum_offset / settings.curriculum_updates, 1.0) if settings.curriculum_updates else 1.0
    env = BatchEnv(args.num_envs, args.seed, profile=args.env_profile, curriculum_progress=initial_curriculum, task_suite=args.task_suite)
    episode_returns = np.zeros(args.num_envs)
    episode_lengths = np.zeros(args.num_envs)
    start_time = time.perf_counter()
    log = (out / "train.jsonl").open("a", encoding="utf-8")
    try:
        for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
            curriculum = min((args.curriculum_offset + iteration - start_iteration) / settings.curriculum_updates, 1.0) if settings.curriculum_updates else 1.0
            env.set_curriculum(curriculum)
            buffer = {k: [] for k in ["obs", "priv", "raw", "logp", "value", "next_value", "reward", "term", "timeout"]}
            completed_returns, completed_lengths, errors, yaw_errors, action_deltas = [], [], [], [], []
            falls = 0
            terrain_levels = []
            terrain_level_by_kind = {}
            saturated, action_count = 0, 0
            for _ in range(args.steps):
                obs, priv = tensor(env.obs.copy(), device), tensor(env.privileged(), device)
                with torch.no_grad():
                    action, raw, logp, value = policy.sample(obs, priv)
                saturated += int((action.abs() > 0.98).sum())
                action_count += action.numel()
                _, rewards, term, timeout, final_obs, final_priv, infos = env.step(action.cpu().numpy())
                with torch.no_grad():
                    next_value = policy.value(tensor(final_obs, device), tensor(final_priv, device))
                for k, v in dict(obs=obs.cpu().numpy(), priv=priv.cpu().numpy(), raw=raw.cpu().numpy(), logp=logp.cpu().numpy(), value=value.cpu().numpy(), next_value=next_value.cpu().numpy(), reward=rewards, term=term, timeout=timeout).items():
                    buffer[k].append(v)
                episode_returns += rewards
                episode_lengths += 1
                errors.extend(i["velocity_error"] for i in infos)
                yaw_errors.extend(i["yaw_error"] for i in infos)
                action_deltas.extend(i["action_delta"] for i in infos)
                terrain_levels.extend(i["terrain_level"] for i in infos if "terrain_level" in i and i["terrain"] != "flat")
                for info in infos:
                    if "terrain_level" in info:
                        terrain_level_by_kind.setdefault(info["terrain"], []).append(info["terrain_level"])
                falls += int(term.sum())
                for i in np.flatnonzero(term | timeout):
                    completed_returns.append(float(episode_returns[i]))
                    completed_lengths.append(float(episode_lengths[i]))
                    episode_returns[i] = episode_lengths[i] = 0
            arrays = {k: np.asarray(v) for k, v in buffer.items()}
            advantages, returns = gae(arrays["reward"], arrays["value"], arrays["next_value"], arrays["term"], arrays["timeout"])
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            batch = {k: tensor(v.reshape((-1,) + v.shape[2:]), device) for k, v in arrays.items() if k in ["obs", "priv", "raw", "logp"]}
            batch["adv"] = tensor(advantages.ravel(), device)
            batch["returns"] = tensor(returns.ravel(), device)
            count = args.steps * args.num_envs
            kls, losses, gaussian_entropies, action_entropies, gradient_norms, clip_fractions = [], [], [], [], [], []
            for _ in range(args.epochs):
                order = torch.randperm(count, device=device)
                epoch_kls = []
                for idx in order.split(args.batch_size):
                    dist = policy.distribution(batch["obs"][idx], batch["priv"][idx])
                    raw = batch["raw"][idx]
                    logp = (dist.log_prob(raw) - torch.log(1. - torch.tanh(raw).square() + 1e-6)).sum(-1)
                    logratio = logp - batch["logp"][idx]
                    ratio = logratio.exp()
                    surrogate = torch.minimum(ratio * batch["adv"][idx], ratio.clamp(0.8, 1.2) * batch["adv"][idx])
                    value_loss = (policy.value(batch["obs"][idx], batch["priv"][idx]) - batch["returns"][idx]).square().mean()
                    gaussian_entropy = dist.entropy().sum(-1).mean()
                    noise = torch.randn(dist.mean.shape, generator=entropy_rng, device=device)
                    action_entropy = squashed_entropy(dist, noise).mean()
                    entropy = gaussian_entropy if args.entropy_mode == "gaussian" else action_entropy
                    loss = -surrogate.mean() + 0.5 * value_loss - args.entropy_coefficient * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                    optimizer.step()
                    approx_kl = float(((ratio - 1) - logratio).mean().detach())
                    epoch_kls.append(approx_kl)
                    kls.append(approx_kl)
                    losses.append(float(loss.detach()))
                    gaussian_entropies.append(float(gaussian_entropy.detach()))
                    action_entropies.append(float(action_entropy.detach()))
                    gradient_norms.append(float(gradient_norm))
                    clip_fractions.append(float(((ratio - 1).abs() > 0.2).float().mean().detach()))
                if np.mean(epoch_kls) > 0.03:
                    break
            total_steps += count
            elapsed = time.perf_counter() - start_time
            row = dict(iteration=iteration, total_steps=total_steps, elapsed_s=elapsed, steps_per_s=(iteration-start_iteration)*count/elapsed, reward=float(arrays["reward"].mean()), velocity_error=float(np.mean(errors)), episode_return=float(np.mean(completed_returns)) if completed_returns else None, episode_length=float(np.mean(completed_lengths)) if completed_lengths else None, loss=float(np.mean(losses)), approx_kl=float(np.mean(kls)))
            row.update(curriculum_progress=curriculum, yaw_error=float(np.mean(yaw_errors)), action_delta=float(np.mean(action_deltas)), falls=falls)
            if terrain_levels:
                row["terrain_level_mean"] = float(np.mean(terrain_levels))
            if terrain_level_by_kind:
                row["terrain_levels"] = {k: dict(min=min(v), max=max(v), samples=len(v)) for k, v in terrain_level_by_kind.items()}
            return_variance = float(np.var(returns))
            row.update(gaussian_entropy=float(np.mean(gaussian_entropies)), action_entropy=float(np.mean(action_entropies)),
                       action_saturation=saturated / action_count, gradient_norm=float(np.mean(gradient_norms)),
                       clip_fraction=float(np.mean(clip_fractions)),
                       action_std=float(policy.log_std.clamp(-3., 1.).exp().mean().detach()),
                       value_explained_variance=1.0 - float(np.var(returns - arrays["value"])) / return_variance if return_variance > 1e-8 else None)
            if not all(np.isfinite(row[k]) for k in ["loss", "reward", "approx_kl"]):
                raise FloatingPointError(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if iteration % 10 == 0 or iteration == start_iteration + 1:
                print(json.dumps(row), flush=True)
            if iteration % args.save_every == 0 or iteration == start_iteration + args.iterations:
                save_checkpoint(out / f"model_{iteration:06d}.pt", policy, optimizer, args, iteration, total_steps)
                save_checkpoint(out / "latest.pt", policy, optimizer, args, iteration, total_steps)
    finally:
        log.close()
        env.close()


if __name__ == "__main__":
    main()
