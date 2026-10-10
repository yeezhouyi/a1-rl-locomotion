"""Read back final fourth-round evidence, figures, page and local links."""
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fourth_round"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths += [v for k, v in attrs if k in ("src", "href") and v]


def main():
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    results = read(OUT / "results.json")
    assert results["episodes"] == 432 and results["seeds"] == [0, 1, 2]
    rows = list(csv.DictReader((OUT / "tables/all_episodes.csv").open(encoding="utf-8-sig", newline="")))
    assert len(rows) == 432
    page = (ROOT / "portfolio/fourth.html").read_text(encoding="utf-8")
    assert page.count('data-method="') == 144
    for method, item in results["overall"].items():
        matched = [r for r in rows if r["method"] == method]
        assert len(matched) == 72
        assert sum(r["tracking_success"] == "True" for r in matched) == item["successes"]
        assert sum(r["fell"] == "True" for r in matched) == item["falls"]
    links = 0
    for filename in ["portfolio/index.html", "portfolio/fourth.html", "docs/fourth_results.md", "docs/round4_plan.md", "docs/resume_project.md", "README.md"]:
        file = ROOT / filename
        text = file.read_text(encoding="utf-8")
        parser = Links()
        parser.feed(text)
        paths = parser.paths if file.suffix == ".html" else re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
        for value in paths:
            if value.startswith(("http:", "https:", "#", "data:")):
                continue
            assert (file.parent / value.split("#")[0]).exists(), (filename, value)
            links += 1
    review = read(OUT / "visual_review.json")
    assert review["figures_checked"] and review["page_checked"] and review["filter_checked"]
    evidence = dict(status="passed", episodes=432, checked_links=links, review=review)
    (OUT / "delivery_audit.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = set(p for p in OUT.rglob("*") if p.is_file() and p.name != "artifact_manifest.json" and "__pycache__" not in p.parts)
    for folder in ["src", "configs", "tools", "tests", "docs", "portfolio", "outputs/fourth_diagnostics_final"]:
        paths.update(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.add(ROOT / "README.md")
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    (OUT / "artifact_manifest.json").write_text(json.dumps(dict(files=hashes), indent=2), encoding="utf-8")
    print(json.dumps(dict(**evidence, hashed_files=len(hashes)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
