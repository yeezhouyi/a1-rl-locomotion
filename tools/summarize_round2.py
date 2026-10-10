"""汇总第二轮开发集/确认集评估 JSONL（仅标准库）。"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "second_round"
TABLES = OUT / "tables"
FIELDS = ["label", "scenario", "n", "velocity_rmse_mean", "velocity_rmse_std",
          "yaw_rmse_mean", "falls", "completed", "success", "mean_steps",
          "action_delta_mean", "power_mean", "recovered_count"]


def load(kind: str):
    rows = []
    for path in sorted(OUT.glob(("dev_*" if kind == "dev" else "confirm_*") + "/episodes.jsonl")):
        label = path.parent.name
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    item = json.loads(line)
                    item["label"] = label
                    item["_path"] = str(path)
                    item["_line"] = line_no
                    rows.append(item)
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["label"], r["scenario"]), []).append(r)
    out = []
    for (label, scenario), rs in sorted(groups.items()):
        n = len(rs)
        def vals(key): return [r[key] for r in rs if r.get(key) is not None]
        def mean(key):
            x = vals(key)
            return statistics.mean(x) if x else ""
        v = vals("velocity_rmse")
        out.append({"label": label, "scenario": scenario, "n": n,
                    "velocity_rmse_mean": statistics.mean(v) if v else "",
                    "velocity_rmse_std": statistics.stdev(v) if len(v) > 1 else (0.0 if v else ""),
                    "yaw_rmse_mean": mean("yaw_rmse"),
                    "falls": sum(bool(r.get("fell", False)) for r in rs),
                    "completed": sum(bool(r.get("completed", False)) for r in rs),
                    "success": sum(bool(r.get("tracking_success", False)) for r in rs),
                    "mean_steps": mean("steps"), "action_delta_mean": mean("action_delta"),
                    "power_mean": mean("power"),
                    "recovered_count": sum(r.get("recovery_s") is not None for r in rs)})
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def write_md(path, kind, rows):
    title = "开发集" if kind == "dev" else "确认集"
    lines = [f"# 第二轮{title}指标", "", "单训练种子，分母为该场景实际评估回合数，±为回合间样本标准差。训练预算、随机化范围与奖励见对应 config；本结果为仿真数据，无真机验证。速度误差仅统计存活阶段，须结合跌倒和存活时间解释。", "", "| 场景 | 策略 | 速度RMSE±std | yawRMSE | 跌倒数 | 成功数 | 存活秒数 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        sec = r["mean_steps"] / 50 if isinstance(r["mean_steps"], (int, float)) else ""
        lines.append(f"| {r['scenario']} | {r['label']} | {fmt(r['velocity_rmse_mean'])}±{fmt(r['velocity_rmse_std'])} | {fmt(r['yaw_rmse_mean'])} | {r['falls']}/{r['n']} | {r['success']}/{r['n']} | {fmt(sec)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dev, confirm = load("dev"), load("confirm")
    if not dev:
        raise SystemExit("未找到 outputs/second_round/dev_*/episodes.jsonl；不生成伪造汇总。")
    def check(rows, kind, count, seed):
        config_name = "validation_v2.json" if kind == "dev" else "confirmation_v2.json"
        scenarios = {r["name"] for r in json.loads((ROOT / "configs" / config_name).read_text(encoding="utf-8"))}
        labels = {r["label"] for r in rows}
        required = {f"{kind}_{p}" for p in ["teacher_original", "baseline_original", "teacher_v2", "baseline_v2"]}
        if kind == "confirm":
            required |= {"confirm_rma_original", "confirm_rma_v2", "confirm_zero_v2"}
            if "confirm_rma_warmstart" in labels:
                required |= {"confirm_teacher_warmstart", "confirm_rma_warmstart", "confirm_zero_warmstart"}
        if not required <= labels:
            raise ValueError(f"Missing labels: {required - labels}")
        for label in labels:
            subset = [r for r in rows if r["label"] == label]
            if {r["scenario"] for r in subset} != scenarios:
                raise ValueError(f"Missing scenarios: {label}")
            for scenario in scenarios:
                seeds = sorted(r["eval_seed"] for r in subset if r["scenario"] == scenario)
                if seeds != list(range(seed, seed + count)):
                    raise ValueError(f"Unpaired/incomplete episodes: {label}/{scenario}")
    check(dev, "dev", 3, 11000)
    write_csv(TABLES / "dev_comparison.csv", summarize(dev)); write_md(ROOT / "docs" / "round2_dev_metrics.md", "dev", summarize(dev))
    if confirm:
        check(confirm, "confirm", 5, 31000)
        c = summarize(confirm); write_csv(TABLES / "confirmation_comparison.csv", c); write_md(ROOT / "docs" / "round2_metrics.md", "confirm", c)


if __name__ == "__main__": main()
