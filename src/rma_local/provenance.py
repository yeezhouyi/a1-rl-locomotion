from pathlib import Path
import hashlib
import platform

import mujoco
import numpy as np
import torch


def provenance():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "configs").glob("*.json"))
    asset_manifest = root / "assets" / "unitree_a1" / "manifest.json"
    if asset_manifest.exists():
        paths.append(asset_manifest)
    return dict(python=platform.python_version(), platform=platform.platform(), mujoco=mujoco.__version__, torch=torch.__version__, numpy=np.__version__, file_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
