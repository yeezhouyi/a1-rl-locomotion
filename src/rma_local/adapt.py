"""Train history-to-latent estimator on rollouts of the deployed student.

Teacher actor and encoder stay frozen. Privileged parameters only form the
regression target; the student controls the simulator with its own estimate.
"""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

from .env import BatchEnv, OBS_DIM
from .models import make_adaptation, Policy, shift_history
from .train import tensor
from .provenance import provenance
from .settings import settings_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--out", default="outputs/adaptation_s0")
    parser.add_argument("--history", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--architecture", choices=["pooled_cnn", "ordered_mlp"], default="pooled_cnn")
    args = parser.parse_args()
    if args.history < 7 or min(args.num_envs, args.steps, args.iterations) <= 0:
        parser.error("history >= 7 and positive counts required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    teacher_ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    if teacher_ckpt["config"]["kind"] != "teacher":
        raise ValueError("Adaptation requires a privileged teacher")
    args.env_profile = teacher_ckpt["config"].get("env_profile", "legacy")
    args.task_suite = teacher_ckpt["config"].get("task_suite", "flat")
    teacher = Policy("teacher").to(device)
    teacher.load_state_dict(teacher_ckpt["policy"])
    teacher.eval().requires_grad_(False)
    student = make_adaptation(args.architecture, args.history).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4)
    env = BatchEnv(args.num_envs, args.seed + 10000, profile=args.env_profile, task_suite=args.task_suite)
    history = torch.zeros(args.num_envs, args.history, OBS_DIM, device=device)
    history[:, -1] = tensor(env.obs, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(dict(config=vars(args), task=settings_dict(args.env_profile), provenance=provenance()), indent=2), encoding="utf-8")
    try:
        with (out / "train.jsonl").open("w", encoding="utf-8") as log:
            for iteration in range(1, args.iterations + 1):
                losses, errors, target_spreads = [], [], []
                for _ in range(args.steps):
                    target = teacher.encoder(tensor(env.privileged(), device)).detach()
                    estimate = student(history)
                    loss = (estimate - target).square().mean()
                    with torch.no_grad():
                        action = teacher.deterministic(tensor(env.obs, device), latent=estimate.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
                    optimizer.step()
                    obs, _, term, timeout, _, _, infos = env.step(action.cpu().numpy())
                    history = shift_history(history, tensor(obs, device), torch.as_tensor(term | timeout, device=device))
                    losses.append(float(loss.detach()))
                    target_spreads.append(float(target.std(dim=0, unbiased=False).mean()))
                    errors.extend(i["velocity_error"] for i in infos)
                row = dict(iteration=iteration, latent_mse=float(np.mean(losses)), target_latent_std=float(np.mean(target_spreads)), velocity_error=float(np.mean(errors)))
                if not np.isfinite(row["latent_mse"]):
                    raise FloatingPointError(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if iteration % 10 == 0 or iteration == 1:
                    print(json.dumps(row), flush=True)
                if iteration % 50 == 0 or iteration == args.iterations:
                    checkpoint = dict(adaptation=student.state_dict(), policy=teacher_ckpt["policy"], config=vars(args), iteration=iteration)
                    torch.save(checkpoint, out / f"model_{iteration:06d}.pt")
                    torch.save(checkpoint, out / "latest.pt")
    finally:
        env.close()


if __name__ == "__main__":
    main()
