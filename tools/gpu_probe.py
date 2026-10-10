"""Measure an actual synchronized GPU batch; compilation is timed separately."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import jax
import jax.numpy as jp
import numpy as np
from gpu_a1_env import A1Joystick


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--impl", default="warp")
    p.add_argument("--terrain", default="flat")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    metadata = dict(backend=jax.default_backend(), devices=[str(x) for x in jax.devices()], batch=args.batch, impl=args.impl,
                    versions={x: importlib.metadata.version(x) for x in ["jax", "mujoco", "mujoco-mjx", "warp-lang", "brax"]})
    if metadata["backend"] != "gpu":
        raise RuntimeError("GPU backend required")
    print(json.dumps(metadata), flush=True)
    env = A1Joystick(args.assets, impl=args.impl, batch_size=args.batch, terrain=args.terrain)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    t = time.perf_counter()
    state = reset(jax.random.split(jax.random.PRNGKey(0), args.batch))
    jax.block_until_ready(state)
    metadata["reset_compile_s"] = time.perf_counter() - t
    actions = jp.zeros((args.batch, 12))
    t = time.perf_counter(); state = step(state, actions); jax.block_until_ready(state)
    metadata["step_compile_s"] = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(100):
        state = step(state, actions)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - t
    metadata.update(timed_control_steps=100*args.batch, rollout_s=elapsed, control_steps_per_s=100*args.batch/elapsed,
                    finite=bool(np.isfinite(np.asarray(state.data.qpos)).all()), observation_shapes={k:list(v.shape) for k,v in state.obs.items()})
    (out/"probe.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
