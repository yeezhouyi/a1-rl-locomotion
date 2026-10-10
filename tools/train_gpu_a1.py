"""Use upstream Brax PPO and Playground rewards to train the ported A1 task."""
import argparse
import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from gpu_a1_env import A1Joystick
from gpu_course_env import A1Course
from gpu_checkpoint import load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=10_000_000)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restore", default="")
    p.add_argument("--reset-range",type=float,default=0.5)
    p.add_argument("--terrain", default="flat")
    p.add_argument("--reward-profile",choices=["upstream","locomotion_v2"],default="upstream")
    args = p.parse_args()
    if args.minibatch_size * 8 % args.batch:
        p.error("--minibatch-size * 8 must be divisible by --batch")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    if jax.default_backend() != "gpu":
        raise RuntimeError("Training requires an actual GPU backend")
    env = A1Course(args.assets,batch_size=args.batch,reward_profile=args.reward_profile) if args.terrain=='course' else A1Joystick(args.assets, batch_size=args.batch, terrain=args.terrain,reward_profile=args.reward_profile,reset_range=args.reset_range)
    factory = functools.partial(networks.make_ppo_networks, policy_hidden_layer_sizes=(512,256,128),
                                value_hidden_layer_sizes=(512,256,128), policy_obs_key="state", value_obs_key="privileged_state")
    train_args = dict(num_timesteps=args.steps, num_evals=3, num_envs=args.batch, num_eval_envs=16,
        episode_length=1000, action_repeat=1, unroll_length=20, num_minibatches=8, num_updates_per_batch=4,
        batch_size=args.minibatch_size, discounting=.97, learning_rate=3e-4, entropy_cost=.01, normalize_observations=True,
        reward_scaling=1.0, max_grad_norm=1.0, num_resets_per_eval=1)
    config = dict(arguments=vars(args), train=train_args, env=env._config.to_dict(), backend=jax.default_backend(),
        versions={x: importlib.metadata.version(x) for x in ["jax","mujoco","mujoco-mjx","brax","warp-lang"]},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name("gpu_a1_env.py"),Path(__file__).with_name("gpu_course_env.py")]},
        note="Single seed A1 task with inherited observations/reward formulas and configured reward weights/reset distribution; not original RMA. Restore reloads parameters/normalization but restarts optimizer.")
    (out/"config.json").write_text(json.dumps(config, indent=2))
    start = time.perf_counter(); callbacks=[]

    def progress(steps, metrics):
        row=dict(steps=int(steps), elapsed_s=time.perf_counter()-start, metrics={k:float(v) for k,v in metrics.items()})
        callbacks.append(row)
        with (out/"progress.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)

    make_policy,params,_=ppo.train(environment=env, **train_args, network_factory=factory, seed=args.seed,
        progress_fn=progress, wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=args.restore or None, save_checkpoint_path=out/"checkpoints")
    latest=max((out/"checkpoints").iterdir(),key=lambda p:int(p.name))
    restored,_=load_policy(latest,deterministic=True)
    returned=make_policy(params,deterministic=True)
    observations={"state":jp.linspace(-.2,.2,8*48).reshape(8,48),
                  "privileged_state":jp.linspace(-.2,.2,8*123).reshape(8,123)}
    keys=jax.random.split(jax.random.PRNGKey(713),8)
    expected=jax.jit(jax.vmap(returned))(observations,keys)[0]
    actual=jax.jit(jax.vmap(restored))(observations,keys)[0]
    error=float(np.max(np.abs(np.asarray(expected)-np.asarray(actual))))
    (out/"restore_verification.json").write_text(json.dumps(dict(checkpoint=str(latest),
        max_action_absolute_error=error,passed=error<1e-5,observations=8),indent=2))
    if not error<1e-5:raise RuntimeError("Restored actions do not match the final in-memory policy")
    (out/"completion.json").write_text(json.dumps(dict(status="complete",elapsed_s=time.perf_counter()-start,
        requested_steps=args.steps, observed_callbacks=[x["steps"] for x in callbacks]),indent=2))
    print("training_complete",flush=True)


if __name__ == "__main__":
    main()
