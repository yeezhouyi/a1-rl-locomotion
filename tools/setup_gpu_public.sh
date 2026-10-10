#!/usr/bin/env bash
set -euo pipefail
task_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_project"
task_reference=outputs/opensource_reference/mujoco_playground
task_commit=d59156e099a8785dd58fbce2e25a6ee1f7b5800c
mkdir -p outputs/opensource_reference
if [[ ! -d "$task_reference" ]]; then
    git init "$task_reference"
    git -C "$task_reference" remote add origin https://github.com/google-deepmind/mujoco_playground.git
    git -C "$task_reference" fetch --depth 1 origin "$task_commit"
    git -C "$task_reference" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$task_reference" rev-parse HEAD)" == "$task_commit" ]]
if [[ ! -e .venv-gpu ]]; then python3.12 -m venv .venv-gpu; fi
export PIP_CACHE_DIR="$task_project/outputs/pip-cache"
.venv-gpu/bin/python -m pip install -e "$task_reference" \
    'jax[cuda12]==0.8.2' 'mujoco==3.15.0' 'mujoco-mjx==3.15.0' \
    'mujoco-warp==3.15.0' 'brax==0.14.2' 'warp-lang==1.18.0'
