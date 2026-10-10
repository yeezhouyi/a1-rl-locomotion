"""Generate round-two plots and a report only from completed local evidence."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/second_round"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if status["stage"] != "confirmation_complete":
        raise RuntimeError("Confirmation incomplete")
    # The summarizer independently checks labels, scenarios, counts and pairing.
    import summarize_round2
    summarize_round2.main()
    labels = ["baseline_original", "teacher_original", "rma_original", "baseline_v2", "teacher_v2", "rma_v2", "zero_v2"]
    warmstart = (OUT / "confirm_rma_warmstart/episodes.jsonl").exists()
    if warmstart:
        if json.loads((OUT / "warmstart_status.json").read_text(encoding="utf-8"))["stage"] != "complete":
            raise RuntimeError("Warmstart branch incomplete")
        labels += ["teacher_warmstart", "rma_warmstart", "zero_warmstart"]
    episodes = {k: read_rows(OUT / f"confirm_{k}/episodes.jsonl") for k in labels}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    training_labels = ["teacher_v2", "baseline_v2"] + (["teacher_from_baseline"] if warmstart else [])
    for kind in training_labels:
        data = read_rows(OUT / kind / "train.jsonl")
        if data[-1]["iteration"] != 7000 or data[-1]["total_steps"] != 1792000:
            raise ValueError("Unexpected PPO budget")
        for ax, key in zip(axes.flat, ["velocity_error", "yaw_error", "action_delta", "falls"]):
            values = np.array([r[key] for r in data])
            smooth = np.convolve(values, np.ones(50) / 50, mode="valid")
            ax.plot([r["total_steps"] - 1280000 for r in data[49:]], smooth, label=kind)
            ax.set(xlabel="Additional control steps", ylabel=key + " (50-update mean)")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "training_v2.png", dpi=160)
    plt.close(fig)
    scenarios = json.loads((ROOT / "configs/confirmation_v2.json").read_text(encoding="utf-8"))
    order = [s["name"] for s in scenarios]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    x = np.arange(len(order))
    chosen = ["rma_original", "baseline_v2", "rma_v2", "rma_warmstart"] if warmstart else ["rma_original", "baseline_v2", "teacher_v2", "rma_v2"]
    for i, label in enumerate(chosen):
        groups = [[r for r in episodes[label] if r["scenario"] == s] for s in order]
        pos = x + (i - 1.5) * 0.19
        axes[0].bar(pos, [np.mean([r["velocity_rmse"] for r in g]) for g in groups], width=0.19, yerr=[np.std([r["velocity_rmse"] for r in g], ddof=1) for g in groups], capsize=2, label=label)
        axes[1].bar(pos, [np.mean([r["steps"] for r in g]) / 50 for g in groups], width=0.19, label=label)
    axes[0].set_ylabel("Velocity RMSE while alive (m/s)")
    axes[1].set_ylabel("Mean survival duration (s)")
    axes[1].set_xticks(x, [s.removeprefix("confirm_") for s in order], rotation=20)
    axes[0].legend(ncol=4)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Frozen confirmation set: one training seed, 5 paired episodes per scenario")
    fig.tight_layout()
    fig.savefig(OUT / "confirmation.png", dpi=160)
    plt.close(fig)
    adapt = read_rows(OUT / "adapt_h50/train.jsonl")
    diagnostics = {k: float(np.mean([r[k] for r in adapt[-50:]])) for k in ["latent_mse", "target_latent_std", "velocity_error"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    adaptation_logs = {"rma_v2": adapt}
    if warmstart:
        adaptation_logs["rma_warmstart"] = read_rows(OUT / "adapt_h50_warmstart/train.jsonl")
    for label, rows in adaptation_logs.items():
        for ax, key in zip(axes, ["latent_mse", "velocity_error"]):
            ax.plot([r["iteration"] for r in rows], [r[key] for r in rows], label=label)
            ax.set(xlabel="Adaptation update", ylabel=key)
            ax.grid(alpha=0.3)
            ax.legend()
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT / "adaptation_v2.png", dpi=160)
    plt.close(fig)
    findings = (OUT / "findings.md").read_text(encoding="utf-8")
    text = ["# RMA 四足自适应：第二轮工程迭代", "", findings, "", "## 固定设计与预算", "", "本轮目标是修正第一轮阶段一的弱电机失稳和偏航误差。完整计划在训练前保存于 [第二轮计划](round2_plan.md)。这是一轮单种子的工程组合迭代，包含额外采样、扩大随机化、课程过渡与奖励调整；不能把差异单独归因于 RMA，也不属于独立的多训练种子复现。", "", "- 教师和基线分别从第一轮各自 128 万步检查点续训，新增各 51.2 万步，累计各 179.2 万控制步；学习率 1e-4，终点为 7,000 次更新。", "- 新范围：摩擦 0.25–1.0、载荷 0–3 kg、电机强度 0.65–1.15；前 1,000 次新增更新线性扩大重置分布，之后完整范围训练。", "- 偏航奖励权重 0.5→1.0，动作差分惩罚 0.08→0.16，跌倒惩罚 2→5；物理、控制、观测和网络未改变。", f"- 新 H50 适应器 {len(adapt)} 次更新、{len(adapt)*256:,} 额外控制步；新 RMA 总采样量为 {1792000+len(adapt)*256:,} 步。旧 RMA 为 1,408,000 步，不能按相同预算比较两版本。", "- 开发集 6 场景×3 回合，种子 11000–11002。确认集在训练前固定，为 9 场景×5 回合，种子 31000–31004；7 组策略共 315 回合。确认集未用于选择检查点。", "- Windows、MuJoCo、A1 模型与 CPU 训练沿用第一轮；无实体机器人和 GPU 训练。", "", "## 指标与原始表格", "", "跟踪成功要求完整回合 20 秒、平面速度 RMSE < 0.2 m/s、偏航 RMSE < 0.3 rad/s。速度误差只统计存活阶段，必须结合跌倒和时长，不能把早跌倒导致的短轨迹误差与完整回合误差等同解释。", "", "| 策略 | 确认集跟踪成功 | 跌倒 | 完整回合 |", "|---|---:|---:|---:|"]
    for label, data in episodes.items():
        text.append(f"| {label} | {sum(r['tracking_success'] for r in data)}/{len(data)} | {sum(r['fell'] for r in data)}/{len(data)} | {sum(r['completed'] for r in data)}/{len(data)} |")
    if warmstart:
        text += ["", "另增加了 teacher_from_baseline 初始化对照及其 H50 适应器，分别为 teacher_warmstart / rma_warmstart。教师从第一轮基线而非第一轮教师初始化，新增预算相同；初始化时将未使用的隐变量输入权重清零，检查初始演员和评论家输出精确等同于基线。它不是新增独立训练种子，也不是原论文规定的初始化。", "", "新增适应器同样训练 750 更新 / 192,000 步；每个新 RMA 分支总采样量均为 1,984,000 步。10 组策略（含初始化对照的零隐变量消融）的确认集总计 450 回合。初始化对照在读取确认成绩前根据开发集问题提出，终点固定为 7,000 更新；验收记录见 `outputs/second_round/warmstart_review.json`。", "", "[从基线初始化后的弱电机同步视频](../outputs/second_round/videos/warmstart_confirm_weak_motor.mp4) / [对应名义场景视频](../outputs/second_round/videos/warmstart_confirm_nominal.mp4)。", ""]
        warm_data = adaptation_logs["rma_warmstart"]
        warm_diagnostics = {key: np.mean([r[key] for r in warm_data[-50:]]) for key in ["latent_mse", "target_latent_std", "velocity_error"]}
        text += [f"初始化对照适应器最后 50 次更新：latent MSE {warm_diagnostics['latent_mse']:.6f}、目标隐变量标准差均值 {warm_diagnostics['target_latent_std']:.6f}、在线速度误差 {warm_diagnostics['velocity_error']:.3f} m/s。须结合 zero_warmstart 的实际控制结果判断隐变量是否带来优势。", ""]
    text += ["", "[逐场景确认集指标](round2_metrics.md) / [开发集指标](round2_dev_metrics.md) / [确认集 CSV](../outputs/second_round/tables/confirmation_comparison.csv)。回合间标准差反映初始关节扰动差异，不能替代跨训练种子的置信区间。", "", "![确认集误差与存活时间](../outputs/second_round/confirmation.png)", "", "## 训练与适应器诊断", "", "![新增阶段一训练曲线](../outputs/second_round/training_v2.png)", "", f"适应器最后 50 次更新均值：latent MSE {diagnostics['latent_mse']:.6f}、目标隐变量维度标准差均值 {diagnostics['target_latent_std']:.6f}、在线速度误差 {diagnostics['velocity_error']:.3f} m/s。这些来自训练轨迹；不能用低 MSE 替代真实控制验证。", "", "![适应器训练](../outputs/second_round/adaptation_v2.png)", "", "## 运动回放", "", "[名义场景同步视频](../outputs/second_round/videos/confirm_nominal.mp4) / [弱电机场景同步视频](../outputs/second_round/videos/confirm_weak_motor.mp4)，从左到右为新基线、特权教师、新 RMA；同一确认种子，提前结束时显示结束标记，不循环回放。", "", "## 复现与证据边界", "", "源码冻结在 `outputs/source_snapshots/round2_training`，其 `snapshot_manifest.json` 保存训练实现与固定配置哈希。原始检查点、每回合 JSONL、有效任务配置、各阶段控制台日志、开发验收记录及本轮证据清单均在 `outputs/second_round`。", "", "旧检查点默认使用 legacy 奖励与控制；新版配置显式记录 robust_v2，并自动传递给适应器与评估。奖励变化后不直接比较跨版本回报。当前 6 项契约测试通过，其中 2 项为本轮新增，旧基线固定回合的物理指标与第一轮一致。", "", "第一轮源文件的原始哈希对应 `outputs/source_snapshots/round1_final` 中的同路径副本；当前工作区源码已更新，不能再用第一轮哈希要求当前文件不变。第一轮日志和评估结果没有覆盖。", "", "确认集大部分参数在扩展后的训练分布内，测试的是固定组合和命令表现；只有 confirm_extrapolation 的摩擦 0.22、载荷 3.3 kg、强度 0.62 超出新范围。若据本轮确认结果继续调参，这一集合也须标为已使用，下一轮另设保留集合。后续仍需独立训练种子、噪声与延迟评估，才能支持稳定泛化的结论。", "", "## 复跑入口", "", "```powershell", ".venv/Scripts/python.exe -m rma_local.evaluate --checkpoint outputs/second_round/adapt_h50/latest.pt --mode rma --scenarios configs/confirmation_v2.json --seed 31000 --episodes 5 --out outputs/recheck_round2_rma", "```", "", "[项目 README](../README.md) / [第一轮报告](first_results.md) / [本轮文件哈希](../outputs/second_round/artifact_manifest.json)", ""]
    report = ROOT / "docs/second_results.md"
    report.write_text("\n".join(text), encoding="utf-8")
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"]
    files += [report, ROOT / "README.md", ROOT / "docs/round2_plan.md", ROOT / "docs/round2_dev_metrics.md", ROOT / "docs/round2_metrics.md"]
    files += list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").glob("*.json")) + [ROOT / "tools/build_round2_report.py", ROOT / "tools/summarize_round2.py", ROOT / "tools/finish_round2.py", ROOT / "tools/warmstart_teacher.py", ROOT / "tools/finish_warmstart.py", ROOT / "outputs/source_snapshots/round2_training/snapshot_manifest.json"]
    manifest = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
