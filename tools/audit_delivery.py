"""Check the local portfolio's evidence, links, assets and final file hashes."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import imageio.v2 as iio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/third_round"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value:
                self.paths.append(value)


def main():
    status = json.loads((OUT / "delivery_status.json").read_text(encoding="utf-8"))
    assert status["stage"] in ["ready_for_final_review", "complete"]
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    assert results["episodes"] == 1155 and results["training_seeds"] == [0, 1, 2]
    raw = (ROOT / "portfolio/data.js").read_text(encoding="utf-8").strip()
    assert raw.startswith("window.ROUND3_DATA = ") and raw.endswith(";")
    view = json.loads(raw.removeprefix("window.ROUND3_DATA = ")[:-1])
    assert view["status"] == "complete" and view["total_episodes"] == 1155
    assert len(view["methods"]) == 7 and len(view["scenario_rows"]) == 77
    for method in view["methods"]:
        actual = results["overall"][method["method"]]
        assert sum(method["success_by_seed"]) == actual["successes"]
        assert sum(method["falls_by_seed"]) == actual["falls"]
        assert abs(method["success_mean_pct"] - actual["mean"]["tracking_success"]*100) < 1e-10
    checked_links = 0
    parser = Links()
    parser.feed((ROOT / "portfolio/index.html").read_text(encoding="utf-8"))
    script_links = re.findall(r'(?:href|src)="([^"$]+)"', (ROOT/'portfolio/app.js').read_text(encoding='utf-8'))
    for value in parser.paths + script_links + [v["src"] for v in view["videos"]]:
        if value.startswith(("http:", "https:", "#", "data:")):
            continue
        assert (ROOT / "portfolio" / value.split("#")[0]).exists(), value
        checked_links += 1
    # Generated third-round report and user-facing entry points must have working local links.
    for file in [ROOT / "docs/third_results.md", ROOT / "README.md", ROOT / 'docs/resume_project.md', ROOT / 'docs/interview_guide.md']:
        for value in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
            if value.startswith(("http:", "https:", "#")):
                continue
            assert (file.parent / value.split("#")[0]).exists(), f"{file.name}: {value}"
            checked_links += 1
    asset_lock = json.loads((ROOT / "assets-lock.json").read_text(encoding="utf-8"))
    for name, record in asset_lock.items():
        assert hashlib.sha256((ROOT / "assets/unitree_a1" / name).read_bytes()).hexdigest() == record["sha256"]
    videos = {}
    for name in ["nominal", "friction_switch", "extrapolation"]:
        path = OUT / "videos" / (name + ".mp4")
        reader = iio.get_reader(path)
        try:
            metadata = reader.get_meta_data()
            first = reader.get_data(0)
            assert metadata["fps"] == 25 and tuple(metadata["size"]) == (2560, 480)
            assert first.shape == (480, 2560, 3)
            videos[name] = dict(size=metadata["size"], fps=metadata["fps"], duration=metadata.get("duration"), bytes=path.stat().st_size)
        finally:
            reader.close()
    latency = results["deployment_latency"]
    for label, folder in [("baseline", "baseline"), ("cnn", "pooled_cnn"), ("ordered", "ordered_mlp")]:
        digest = hashlib.sha256((OUT / "seed_0" / folder / "latest.pt").read_bytes()).hexdigest()
        assert latency["results"][label]["checkpoint_sha256"] == digest
    deployment = json.loads((OUT / 'deployment/manifest.json').read_text(encoding='utf-8'))
    for label, item in deployment['models'].items():
        filename = OUT / 'deployment' / item['file']
        assert hashlib.sha256(filename.read_bytes()).hexdigest() == item['sha256']
        assert item['max_abs_difference'] <= 1e-6
        assert item['source_sha256'] == latency['results'][label]['checkpoint_sha256']
    review = json.loads((OUT / "visual_review.json").read_text(encoding="utf-8"))
    assert review["figures_checked"] and review["video_frames_checked"] and review["page_checked"] and review["filter_checked"]
    evidence = dict(episodes=1155, training_seeds=3, strategy_groups=7, checked_links=checked_links,
                    asset_files=len(asset_lock), videos=videos, visual_review=review, status="passed")
    (OUT / "delivery_audit.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    # Final manifest records current reporting/UI code as well as the frozen training source.
    files = set(p for p in OUT.rglob("*") if p.is_file() and p.name != "artifact_manifest.json")
    for folder in ["src", "configs", "tools", "docs", "portfolio", "tests", "outputs/source_snapshots/round3_training"]:
        files.update(p for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.update(ROOT/p for p in ["README.md", "pyproject.toml", "requirements-lock.txt", "assets-lock.json"])
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (OUT / "artifact_manifest.json").write_text(json.dumps(dict(files=hashes, note="Hashes generated after final evidence and UI review; training source version is identified separately in source_snapshot.json."), indent=2), encoding="utf-8")
    print(json.dumps(dict(status="passed", hashed_files=len(hashes), checked_links=checked_links, videos=videos), indent=2))


if __name__ == "__main__":
    main()
