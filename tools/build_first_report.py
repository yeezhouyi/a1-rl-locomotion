"""Build the Chinese first-run report from completed, unmodified evaluations."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/first_comparison")
    args = parser.parse_args()
    run = ROOT / args.run
    labels = {"baseline": "域随机化基线", "oracle": "特权策略", "rma_h50": "RMA H50", "rma_h10": "RMA H10", "zero_latent": "零隐变量"}
    names = {"nominal": "名义条件", "heldout_low_friction": "低摩擦", "heldout_payload": "载荷", "heldout_weak_motor": "弱电机", "heldout_combination": "组合扰动", "friction_switch": "摩擦突变", "lateral_push": "侧向推扰"}
    rows = {label: jsonl(run / f"eval_{label}" / "episodes.jsonl") for label in labels}
    # A partial run must never generate a report that looks complete.
    expected_scenarios = {r["name"] for r in json.loads((ROOT / "configs/eval_scenarios.json").read_text(encoding="utf-8"))}
    for label, data in rows.items():
        if len(data) != 21 or {r["scenario"] for r in data} != expected_scenarios:
            raise ValueError(f"Incomplete evaluation: {label}")
        for scenario in expected_scenarios:
            seeds = sorted(r["eval_seed"] for r in data if r["scenario"] == scenario)
            if seeds != [20000, 20001, 20002]:
                raise ValueError(f"Unpaired seeds: {label}/{scenario}")
    fingerprints = [json.loads((run / f"eval_{label}/summary.json").read_text(encoding="utf-8"))["provenance"]["file_sha256"] for label in labels]
    if any(p != fingerprints[0] for p in fingerprints):
        raise ValueError("Evaluation source/assets differ")
    budgets = {}
    for kind in ["teacher", "baseline"]:
        ckpt = torch.load(ROOT / f"outputs/{kind}_5000_s0/model_005000.pt", map_location="cpu", weights_only=False)
        config = ckpt["config"]
        budgets[kind] = ckpt["iteration"] * config["num_envs"] * config["steps"]
        if ckpt["iteration"] != 5000 or budgets[kind] != 1280000:
            raise ValueError(f"Unexpected budget: {kind}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    diagnostics = {}
    for h in [50, 10]:
        data = jsonl(run / f"adapt_h{h}" / "train.jsonl")
        ckpt = torch.load(run / f"adapt_h{h}/latest.pt", map_location="cpu", weights_only=False)
        if ckpt["iteration"] != 500 or len(data) != 500:
            raise ValueError(f"Incomplete adaptation: H{h}")
        config = ckpt["config"]
        diagnostics[h] = {key: float(np.mean([r[key] for r in data[-50:]])) for key in ["latent_mse", "target_latent_std", "velocity_error"]}
        diagnostics[h]["steps"] = ckpt["iteration"] * config["num_envs"] * config["steps"]
        for ax, key in zip(axes, ["latent_mse", "target_latent_std", "velocity_error"]):
            ax.plot([r["iteration"] for r in data], [r[key] for r in data], alpha=0.8, label=f"H{h}")
            ax.set(xlabel="Adaptation update", ylabel=key)
            ax.grid(alpha=0.3)
            ax.legend()
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(run / "adaptation.png", dpi=160)
    plt.close(fig)
    scenario_order = [r["name"] for r in json.loads((ROOT / "configs/eval_scenarios.json").read_text(encoding="utf-8"))]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    x = np.arange(len(scenario_order))
    width = 0.15
    for j, (label, data) in enumerate(rows.items()):
        groups = [[r for r in data if r["scenario"] == s] for s in scenario_order]
        positions = x + (j - 2) * width
        errors = [np.mean([r["velocity_rmse"] for r in g]) for g in groups]
        stds = [np.std([r["velocity_rmse"] for r in g], ddof=1) for g in groups]
        axes[0].bar(positions, errors, width=width, yerr=stds, capsize=2, label=label)
        axes[1].bar(positions, [np.mean([r["steps"] for r in g]) / 50 for g in groups], width=width, label=label)
    axes[0].set_ylabel("Velocity RMSE while alive (m/s)")
    axes[1].set_ylabel("Mean survival duration (s)")
    axes[1].set_xticks(x, [s.replace("heldout_", "") for s in scenario_order], rotation=15)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(ncol=5, fontsize=9)
    fig.suptitle("Single training seed; 3 paired episodes per scenario")
    fig.tight_layout()
    fig.savefig(run / "evaluation.png", dpi=160)
    plt.close(fig)

    def avg(data, key):
        return float(np.mean([r[key] for r in data]))

    text = ["# RMA 四足运动自适应：第一轮本地实验", "", "本轮完成了 RMA 核心方法的独立实现、单训练种子对照、历史长度消融及固定扰动测试。实验结果见下表；是否提升必须结合跟踪误差、跌倒与完整回合判断。这里不将单种子结果解释为稳定性能提升，也不声称复现了原论文全部结论。", "", "## 运行条件与预算", "", "- Windows 原生 MuJoCo 3.3.5，Unitree A1 公开模型，PyTorch 2.8.0+cpu；仿真与网络均使用 CPU。本机 RTX 4060 未参与本轮训练。", "- 教师与域随机化基线各 5,000 次 PPO 更新、1,280,000 控制步；每次更新 8 环境 × 32 步。控制 50 Hz，物理 500 Hz。", "- H50 与 H10 适应器各 500 次更新、128,000 额外控制步，分别对应 1.0 秒和 0.2 秒历史。每个 RMA 分支总采样量为 1,408,000 步。", "- 训练种子均为 0；续训恢复模型和优化器，不恢复物理状态及 RNG，不能逐位重放。", "- 教师验证集 2 场景 × 3 回合；最终测试 7 场景 × 3 回合 × 最多 20 秒，共 105 个策略测试回合，评估种子 20000–20002。", "- 本轮固定采用 5,000 更新的终点检查点，没有根据最终测试成绩挑选历史检查点。验证集此前用于观察训练是否学会运动。", "", "## 任务与信息边界", "", "部署策略只读取 45 维本体观测：局部角速度、重力投影、命令、关节偏差和速度、上次动作。RMA 用观测历史估计 8 维隐变量，不读取摩擦、载荷或执行器强度；特权策略直接编码这些参数，仅作参考。域随机化基线单独训练，其隐变量固定为零。", "", "训练范围为摩擦 0.4–1.0、附加载荷 0–2 kg、执行器强度 0.85–1.15；最终测试包含摩擦 0.25、载荷 3 kg、强度 0.7 及其组合。所有最终场景的前向命令为 0.5 m/s。摩擦在第 300 控制步突变；推扰为第 300 步起持续 0.1 秒的 40 N 侧向力。", "", "## 场景对照", "", "速度 RMSE 单位为 m/s，统计存活期间的二维平面速度误差。成功要求完整存活 20 秒、速度 RMSE < 0.2 m/s、偏航 RMSE < 0.3 rad/s。每个比例只有 3 回合的分辨率；± 为回合间样本标准差，不是跨训练种子置信区间。", "", "| 场景 | 策略 | 速度 RMSE（均值±标准差） | 跌倒 | 跟踪成功 | 平均存活步数 |", "|---|---|---:|---:|---:|---:|"]
    for scenario in [r["name"] for r in json.loads((ROOT / "configs/eval_scenarios.json").read_text(encoding="utf-8"))]:
        for label in labels:
            data = [r for r in rows[label] if r["scenario"] == scenario]
            errors = [r["velocity_rmse"] for r in data]
            text.append(f"| {names.get(scenario, scenario)} | {labels[label]} | {np.mean(errors):.3f} ± {np.std(errors, ddof=1):.3f} | {sum(r['fell'] for r in data)}/3 | {sum(r['tracking_success'] for r in data)}/3 | {avg(data, 'steps'):.0f} |")
    text += ["", "## 摩擦突变与部署开销", "", "恢复时间从摩擦突变时刻到误差连续 0.5 秒满足阈值的区间起点计算；记录为 0 秒表示突变后立即进入达标区间，不能单独证明主动适应。若达标后再次失稳，本指标不会撤销第一次恢复记录，须结合完整回合结果。推扰目前只统计存活和误差。", "", "| 策略 | 摩擦突变恢复回合 | 已恢复回合平均时间（s） | 各回合推理 p95 均值（ms） |", "|---|---:|---:|---:|"]
    for label, data in rows.items():
        switch = [r for r in data if r["scenario"] == "friction_switch"]
        times = [r["recovery_s"] for r in switch if r["recovery_s"] is not None]
        recovery = f"{np.mean(times):.3f}" if times else "—（无恢复）"
        text.append(f"| {labels[label]} | {len(times)}/3 | {recovery} | {avg(data, 'inference_p95_ms'):.3f} |")
    text += ["", "推理延迟包含 Python 与模型前向，排除物理仿真；部分评估期间存在其他训练进程，并非独占 CPU 的严格性能基准。", "", "## 适应器诊断", "", "以下为最后 50 次更新的均值，来自训练轨迹而非独立测试集。", "", "| 历史长度 | latent MSE | 目标隐变量维度标准差均值 | 在线速度误差（m/s） |", "|---|---:|---:|---:|"]
    for h, d in diagnostics.items():
        text.append(f"| H{h} | {d['latent_mse']:.6f} | {d['target_latent_std']:.6f} | {d['velocity_error']:.3f} |")
    text += ["", "MSE 小不能单独证明系统识别成功，目标隐变量波动也不足以证明编码具有物理可辨识性。当前一维卷积局部感受野为 7 帧，再做时间平均池化：H50 与 H10 比较的是输入窗口和统计覆盖，不能解释为模型能够建模整段 50 帧的有序依赖。", "", "![PPO 训练曲线](../outputs/first_comparison/training.png)", "", "![适应器训练诊断](../outputs/first_comparison/adaptation.png)", "", "## 结果使用边界与下一步", "", "本实验为平地、三类隐藏动力学参数和无噪声本体观测的仿真研究；载荷采用躯干中心质量与惯量缩放近似。未包含复杂地形、传感器噪声、真机或原论文的完整环境因子与奖励。测试参数部分超出训练范围，失败是有效的外推结果。", "", "下一步先在独立开发验证集排查教师跟踪质量与适应器误差，再增加训练种子检验趋势；如本轮 RMA 没有优势，应将其作为负结果分析，不能在简历中写性能提升。用最终测试结果指导下一版改动后，应将本轮测试标为已使用的探索性测试，并另设保留集确认结论。", "", "## 原始证据与复跑", "", "- [逐场景汇总 CSV](../outputs/first_comparison/comparison/comparison.csv)，[完整汇总表](../outputs/first_comparison/comparison/comparison.md)。", "- [教师验证结果](../outputs/first_comparison/validation_oracle/summary.json)。各 `eval_*` 目录保存逐回合 JSONL、指标汇总、参数与源码哈希。", "- [H50 名义场景视频](../outputs/first_comparison/videos/nominal.mp4)，[H50 摩擦突变视频](../outputs/first_comparison/videos/friction_switch.mp4)。", "- `outputs/teacher_5000_s0/model_005000.pt`、`outputs/baseline_5000_s0/model_005000.pt` 和本轮 `adapt_h*/latest.pt` 为固定检查点。", "- [实验证据文件哈希](../outputs/first_comparison/artifact_manifest.json)；模型资产来源与哈希见 `assets/unitree_a1/manifest.json`。", "- 安装、独立训练和评估命令见 [README](../README.md)，方法差异见 [方法说明](method.md)。", ""]
    def scenario_mean(label, scenario, key):
        return avg([r for r in rows[label] if r["scenario"] == scenario], key)

    findings = ["## 本轮结论", "", f"- RMA H50 与基线各通过 {sum(r['tracking_success'] for r in rows['rma_h50'])}/21 和 {sum(r['tracking_success'] for r in rows['baseline'])}/21 个跟踪回合；跌倒分别为 {sum(r['fell'] for r in rows['rma_h50'])}/21 与 {sum(r['fell'] for r in rows['baseline'])}/21。优势取决于场景，不能声称全面更鲁棒。", f"- 低摩擦速度 RMSE：H50 {scenario_mean('rma_h50', 'heldout_low_friction', 'velocity_rmse'):.3f}，基线 {scenario_mean('baseline', 'heldout_low_friction', 'velocity_rmse'):.3f} m/s。摩擦突变全回合 RMSE：H50 {scenario_mean('rma_h50', 'friction_switch', 'velocity_rmse'):.3f}，基线 {scenario_mean('baseline', 'friction_switch', 'velocity_rmse'):.3f} m/s。但所有策略都未通过连续 0.5 秒误差达标的恢复判据，不能宣称已实现快速恢复。", "- 3 kg 载荷下，H50 三回合都不跌倒但偏航 RMSE 约 0.377 rad/s，未通过任务标准；基线三回合都通过。电机强度 0.7 下，H50 与教师三回合都迅速跌倒，基线三回合均通过。组合扰动下所有主要策略均失败。教师自身的范围外控制缺陷会限制适应器。", "- H50 名义条件成功 3/3，H10 为 0/3，主要差别是 H10 偏航 RMSE 约 0.304 rad/s，略高于 0.3 阈值；这一二元成功率差异不能解释为显著、稳定的性能差距。", "- 将教师隐变量直接置零会使名义场景迅速跌倒，说明当前教师依赖该输入；零隐变量属于分布改变，不能单独证明适应模块优于独立训练的基线。", "", "这些结果提供了可展示的算法链路、消融和负结果分析；下一轮优先修正阶段一的外推与偏航问题，再增加训练种子。", ""]
    text[4:4] = findings
    text += ["## 视频核对与动作质量", "", "[名义条件三策略同步视频](../outputs/first_comparison/videos/comparison_nominal.mp4)和[摩擦突变三策略同步视频](../outputs/first_comparison/videos/comparison_friction_switch.mp4)按 baseline / oracle / rma_h50 顺序排列，使用同一回合种子。画面采用跟随相机，应结合棋盘地面与日志判断前进；如回合提前结束，合成工具用黑屏和结束标记显示，不循环播放。", "", f"名义条件动作相邻步变化平方均值为：基线 {scenario_mean('baseline', 'nominal', 'action_delta'):.3f}，H50 {scenario_mean('rma_h50', 'nominal', 'action_delta'):.3f}；末步机械功率采样均值约 {scenario_mean('baseline', 'nominal', 'power'):.1f} W 与 {scenario_mean('rma_h50', 'nominal', 'power'):.1f} W。两者都有较大的动作变化；此处不把仿真末步功率采样当成电池能耗，也不声称步态已达到真机部署要求。", "", "![扰动评估与存活时间](../outputs/first_comparison/evaluation.png)", "", "## 版本记录与简历表述", "", "最终五组评估的源码、配置和资产哈希一致。训练过程中曾更新棋盘地面与照明，教师与基线训练配置记录的 env.py 哈希不同；历史阶段没有保存完整源码快照。已保留各阶段配置与日志；下一轮应在固定 Git 版本下从头训练，提高历史实现的可追溯性。", "", "> 基于 PyTorch 与 MuJoCo 实现 Unitree A1 四足机器人 RMA 核心方法，完成 PPO 特权策略、历史隐变量估计及无特权参数部署；开展域随机化基线、历史长度与零隐变量消融，在 7 类固定场景中记录跟踪误差、跌倒率和恢复指标，分析低摩擦改善及弱执行器外推失败。", "", "该表述强调已完成的工作；如列出数值，须注明单训练种子和仿真范围，不能写成真机性能提升。", ""]
    report = ROOT / "docs/first_results.md"
    report.write_text("\n".join(text), encoding="utf-8")
    files = [p for p in run.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"]
    files += [ROOT / f"outputs/{kind}_5000_s0/model_005000.pt" for kind in ["teacher", "baseline"]]
    for directory in ["native_pd_benchmark_s0", "quick_baseline_s0", "baseline_5000_s0", "native_pd_teacher_s0", "fast_teacher_s0", "teacher_5000_s0"]:
        files += [ROOT / "outputs" / directory / "config.json", ROOT / "outputs" / directory / "train.jsonl"]
    files += list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").glob("*.json"))
    files += [report, ROOT / "README.md", ROOT / "requirements-lock.txt", ROOT / "assets/unitree_a1/manifest.json"]
    files += [ROOT / "tools/build_first_report.py", ROOT / "tools/compare_videos.py", ROOT / "tools/finish_local_run.py"]
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (run / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
