"""Fixed-seed, frozen-checkpoint nominal-motion diagnostics; not a benchmark."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rma_local.env import A1Env, rotation
from rma_local.models import Policy
from rma_local.ppo import tanh_log_jacobian


ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "outputs" / "fourth_round_diagnostics"
PARAMS = [0.7, 0.0, 1.0]
COMMAND = [0.5, 0.0, 0.0]
EVAL_SEED = 52000
HORIZON = 1000
MC_SAMPLES = 64


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def diag_policy(policy, obs, priv, generator):
    dist = policy.distribution(obs, priv)
    mean = dist.mean
    eps = torch.randn((MC_SAMPLES,) + tuple(mean.shape), generator=generator)
    raw_samples = mean.unsqueeze(0) + dist.scale.unsqueeze(0) * eps
    log_jac = tanh_log_jacobian(raw_samples)
    gaussian_h = dist.entropy().sum(-1)
    transformed_h = gaussian_h + log_jac.sum(-1).mean(0)
    return torch.tanh(mean), dict(
        mean_abs_raw_mean=float(mean.abs().mean()),
        action_saturation_fraction=float((torch.tanh(mean).abs() > 0.98).float().mean()),
        squash_sensitivity_mean=float((1.0 - torch.tanh(mean).square()).mean()),
        gaussian_entropy=float(gaussian_h.item()),
        true_entropy_mc=float(transformed_h.item()),
    )


def run_one(label, ckpt_path, kind, random_policy, out):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy = Policy(kind)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    profile = ckpt["config"].get("env_profile", "legacy")
    env = A1Env(randomize=False, horizon=HORIZON, profile=profile)
    rows = []
    try:
        obs, _ = env.reset(seed=EVAL_SEED, options={"parameters": PARAMS, "command": COMMAND})
        priv = torch.from_numpy(env.privileged())[None]
        torch_gen = torch.Generator(device="cpu").manual_seed(EVAL_SEED + 900001)
        # Separate local RNG means entropy samples do not consume policy sampling RNG.
        np_gen = np.random.default_rng(EVAL_SEED)
        fell = False
        with torch.inference_mode():
            for step in range(HORIZON):
                ob = torch.from_numpy(obs)[None]
                mean_action, m = diag_policy(policy, ob, priv, torch_gen)
                if random_policy:
                    dist = policy.distribution(ob, priv)
                    noise = torch.from_numpy(np_gen.normal(size=mean_action.shape).astype(np.float32))
                    action_np = torch.tanh(dist.mean + dist.stddev * noise)[0].numpy()
                else:
                    action_np = mean_action[0].numpy()
                obs, reward, term, timeout, info = env.step(action_np)
                v = rotation(env.data) .T @ env.data.qvel[:3]
                row = dict(step=step + 1, true_vx=float(v[0]), true_vy=float(v[1]), true_yaw=float(env.data.qvel[5]),
                           base_height=float(env.data.qpos[2]), velocity_error=float(info["velocity_error"]),
                           reward=float(reward), **m, fell=bool(term))
                rows.append(row)
                if term or timeout:
                    fell = bool(term)
                    break
    finally:
        env.close()
    csv_path = out / f"{label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    keys = ["true_vx", "true_yaw", "base_height", "velocity_error", "reward", "mean_abs_raw_mean",
            "action_saturation_fraction", "squash_sensitivity_mean", "gaussian_entropy", "true_entropy_mc"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update(steps=len(rows), fell=fell, completed=(len(rows) == HORIZON and not fell),
                   checkpoint=str(ckpt_path.resolve()), checkpoint_sha256=sha256(ckpt_path), kind=kind,
                   policy_mode="stochastic_squashed_policy" if random_policy else "deterministic_tanh_mean",
                   profile=profile, csv=str(csv_path.resolve()))
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.out}")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    results = {}
    curves = {}
    provenance = []
    try:
        for seed in range(3):
            for kind in ("baseline", "teacher"):
                cp = ROOT / "outputs" / "third_round" / f"seed_{seed}" / kind / "latest.pt"
                if not cp.is_file():
                    raise FileNotFoundError(cp)
                for rnd in (False, True):
                    label = f"seed_{seed}_{kind}_{'stochastic' if rnd else 'deterministic'}"
                    summary, rows = run_one(label, cp, kind, rnd, args.out)
                    results[label] = summary
                    if not rnd:
                        curves[label] = rows
                provenance.append(dict(path=str(cp.resolve()), sha256=sha256(cp)))
    finally:
        pass
    with (args.out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(dict(parameters=PARAMS, command=COMMAND, eval_seed=EVAL_SEED, horizon=HORIZON,
                       entropy_mc_samples=MC_SAMPLES, source_sha256=sha256(Path(__file__)), note="true_entropy_mc uses 64 independent Gaussian reparameterized samples per step and stable tanh log-Jacobian. Stochastic frozen-policy sampling is diagnostic only, not a formal performance benchmark.",
                       checkpoints=provenance, results=results), f, indent=2)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for label, rows in curves.items():
        x = [r["step"] for r in rows]
        axes[0].plot(x, [r["true_vx"] for r in rows], label=label, alpha=.8)
        axes[1].plot(x, [r["base_height"] for r in rows], label=label, alpha=.8)
        axes[2].plot(x, [r["action_saturation_fraction"] for r in rows], label=label, alpha=.8)
    for ax, title in zip(axes, ["True base vx (m/s)", "Base height (m)", "Action saturation fraction"]):
        ax.set_ylabel(title); ax.grid(True, alpha=.25)
    axes[-1].set_xlabel("Control step")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(args.out / "trajectories.png", dpi=150); plt.close(fig)
    for label, x in results.items():
        if label.endswith("deterministic"):
            print(f"{label}: mean_vx={x['true_vx']:.4f}, gaussian_H={x['gaussian_entropy']:.4f}, transformed_H_MC={x['true_entropy_mc']:.4f}, diff={x['true_entropy_mc']-x['gaussian_entropy']:.4f}, steps={x['steps']}, fell={x['fell']}")
    print(f"outputs={args.out.resolve()}")


if __name__ == "__main__":
    main()
