"""Export a local, self-contained evidence bundle without the virtualenv."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/releases/rma_a1_portfolio.zip")
    parser.add_argument("--include-round4", action="store_true")
    args = parser.parse_args()
    target = ROOT / args.out
    if target.exists():
        raise FileExistsError(target)
    state = json.loads((ROOT / "outputs/third_round/status.json").read_text(encoding="utf-8"))
    if state["stage"] != "confirmation_complete":
        raise RuntimeError("Third-round evidence incomplete")
    paths = set()
    for directory in ["src", "configs", "tools", "docs", "tests", "portfolio", "assets/unitree_a1"]:
        paths.update(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for name in ["README.md", "pyproject.toml", ".gitignore", "requirements-lock.txt", "assets-lock.json"]:
        path = ROOT / name
        if path.exists():
            paths.add(path)
    # Include complete logs and evaluations; only final model weights to keep size bounded.
    directories = ["outputs/third_round", "outputs/second_round", "outputs/first_comparison", "outputs/baseline_5000_s0", "outputs/teacher_5000_s0"]
    if args.include_round4:
        round4 = ROOT / "outputs/fourth_round/delivery_audit.json"
        if not round4.exists() or json.loads(round4.read_text(encoding="utf-8"))["status"] != "passed":
            raise RuntimeError("Fourth-round delivery must pass audit before packaging")
        directories += ["outputs/fourth_round", "outputs/fourth_diagnostics_final"]
    for directory in directories:
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file():
                continue
            if path.suffix == ".pt" and path.name != "latest.pt" and not path.name.endswith('.torchscript.pt'):
                continue
            paths.add(path)
    paths.update(p for p in (ROOT / "outputs/source_snapshots").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    # The artifact manifests name all checkpoints, including ones omitted from this lean bundle.
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("bundle_manifest.json", json.dumps(dict(files=manifest,
            note="Lean local evidence bundle: final latest.pt checkpoints included; numbered checkpoints omitted. Original manifests remain unchanged and may reference omitted intermediate files."), ensure_ascii=False, indent=2))
    # Read every included file back and compare bytes to its original hash.
    with zipfile.ZipFile(target) as archive:
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(name)
    result = dict(path=str(target), files=len(paths), bytes=target.stat().st_size,
                  sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
