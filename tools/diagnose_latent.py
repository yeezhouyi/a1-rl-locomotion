"""Teacher latent sensitivity on held-fixed observations; diagnostic, not performance."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rma_local.env import A1Env
from rma_local.models import Policy
from rma_local.settings import task_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(1)
    torch.manual_seed(24680)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["config"]["kind"] != "teacher":
        raise ValueError("Expected teacher")
    policy = Policy("teacher")
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    profile = checkpoint["config"].get("env_profile", "legacy")
    ranges = np.asarray(task_settings(profile).train_ranges)
    parameters = np.random.default_rng(24680).uniform(ranges[:,0], ranges[:,1], (4096,3))
    normalized = torch.as_tensor((parameters - [0.7, 1, 1]) / [0.3, 1, .15], dtype=torch.float32)
    env = A1Env(randomize=False, horizon=1000, profile=profile)
    observations = []
    try:
        for episode in range(3):
            obs, _ = env.reset(seed=52000+episode, options=dict(parameters=[.7, 1., 1.], command=[.4,0.,0.]))
            for step in range(1000):
                with torch.inference_mode():
                    action = policy.deterministic(torch.from_numpy(obs)[None], torch.from_numpy(env.privileged())[None])[0].numpy()
                if step % 20 == 0:
                    observations.append(obs.copy())
                obs, _, term, timeout, _ = env.step(action)
                if term or timeout:
                    break
    finally:
        env.close()
    obs = torch.from_numpy(np.asarray(observations))
    with torch.inference_mode():
        latents = policy.encoder(normalized)
        constant = latents.mean(0, keepdim=True)
        varied_actions = []
        # Fix each observation, vary the training-distribution latent input.
        for z in latents[::64]:
            varied_actions.append(policy.deterministic(obs, latent=z[None].expand(len(obs), -1)))
        actions = torch.stack(varied_actions)
        constant_actions = policy.deterministic(obs, latent=constant.expand(len(obs),-1))
        zero_actions = policy.deterministic(obs, latent=torch.zeros(len(obs),8))
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), parameter_samples=4096,
                  observations=len(obs), varied_latent_samples=len(varied_actions),
                  latent_std_per_dimension=latents.std(0, unbiased=False).tolist(),
                  action_std_mean=float(actions.std(0,unbiased=False).mean()),
                  action_deviation_from_constant_mean=float((actions-constant_actions).abs().mean()),
                  constant_vs_zero_action_mean_abs=float((constant_actions-zero_actions).abs().mean()),
                  note="Held-observation sensitivity on nominal teacher trajectories; not an online adaptation success metric. Action range [-1,1].")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
