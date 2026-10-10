"""Isolated warmed-up CPU deployment forward latency, excluding simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from rma_local.env import OBS_DIM
from rma_local.models import Policy, make_adaptation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--cnn", required=True)
    parser.add_argument("--ordered", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(1)
    obs = torch.zeros(1, OBS_DIM)
    history = torch.zeros(1, 50, OBS_DIM)
    results = {}
    for label in ["baseline", "cnn", "ordered"]:
        filename = Path(getattr(args,label))
        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        policy = Policy("baseline" if label == "baseline" else "teacher")
        policy.load_state_dict(checkpoint["policy"])
        policy.eval()
        adapter = None
        if label != "baseline":
            adapter = make_adaptation(checkpoint["config"].get("architecture", "pooled_cnn"), 50)
            adapter.load_state_dict(checkpoint["adaptation"])
            adapter.eval()
        elapsed = []
        with torch.inference_mode():
            for i in range(2200):
                start = time.perf_counter_ns()
                if adapter is None:
                    policy.deterministic(obs)
                else:
                    policy.deterministic(obs, latent=adapter(history))
                end = time.perf_counter_ns()
                if i >= 200:
                    elapsed.append((end-start)/1e6)
        results[label] = dict(median_ms=float(np.median(elapsed)), p95_ms=float(np.percentile(elapsed,95)),
                              p99_ms=float(np.percentile(elapsed,99)),
                              parameters=sum(p.numel() for p in policy.actor.parameters()) + (sum(p.numel() for p in adapter.parameters()) if adapter else 0),
                              checkpoint_sha256=hashlib.sha256(filename.read_bytes()).hexdigest())
    result = dict(platform=platform.platform(), processor=platform.processor(), torch=torch.__version__,
                  threads=1, batch=1, warmup=200, samples=2000, results=results,
                  note="Model forward including Python calls on zero tensors; excludes history update, sensors, simulator and control. No hard realtime guarantee.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
