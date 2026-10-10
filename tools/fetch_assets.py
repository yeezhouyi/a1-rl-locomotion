"""Download only the A1 Menagerie model; retain upstream license and hashes."""
from pathlib import Path
import hashlib
import json
import urllib.request

project = Path(__file__).resolve().parents[1]
root = project / "assets" / "unitree_a1"
root.mkdir(parents=True, exist_ok=True)
files = ["a1.xml", "scene.xml", "LICENSE", "README.md", "assets/calf.obj", "assets/hip.obj", "assets/thigh.obj", "assets/thigh_mirror.obj", "assets/trunk.obj", "assets/trunk_A1.png"]
manifest = {}
lock_path = project / "assets-lock.json"
lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
for filename in files:
    target = root / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    url = lock.get(filename, {}).get("source", "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_a1/" + filename)
    if not target.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if filename in lock and digest != lock[filename]["sha256"]:
            raise ValueError(f"Upstream asset differs from frozen experiment: {filename}. Use the included evidence bundle assets.")
        target.write_bytes(data)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if filename in lock and digest != lock[filename]["sha256"]:
        raise ValueError(f"Local asset checksum mismatch: {filename}; existing file left intact.")
    manifest[filename] = {"source": url, "sha256": digest}
    print(filename)
manifest_path = root / "manifest.json"
# Preserve original experiment metadata bytes, including its recorded hashes.
if not manifest_path.exists():
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
elif json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
    raise ValueError("Asset manifest differs; refuse to overwrite experiment evidence.")
