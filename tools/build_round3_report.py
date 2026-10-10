"""Validate three-seed evidence and build tables, figures and a Chinese report."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/third_round"
METHODS = ["baseline", "oracle", "cnn", "ordered", "constant", "zero", "ordered_shuffled"]
LABELS = dict(baseline="域随机化基线", oracle="特权教师", cnn="CNN 适应器", ordered="有序历史 MLP",
              constant="固定隐变量", zero="零隐变量", ordered_shuffled="打乱历史 MLP")
METRICS = ["tracking_success", "fell", "completed", "velocity_rmse", "yaw_rmse", "steps", "action_delta", "power"]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_metrics(group):
    return {key: float(np.mean([row[key] for row in group])) for key in METRICS}


def across_seeds(groups):
    per_seed = {str(seed): seed_metrics(group) for seed, group in groups.items()}
    means = {key: float(np.mean([v[key] for v in per_seed.values()])) for key in METRICS}
    stds = {key: float(np.std([v[key] for v in per_seed.values()], ddof=1)) for key in METRICS}
    return dict(mean=means, std=stds, per_seed=per_seed,
                successes=sum(sum(r["tracking_success"] for r in g) for g in groups.values()),
                falls=sum(sum(r["fell"] for r in g) for g in groups.values()),
                completed=sum(sum(r["completed"] for r in g) for g in groups.values()),
                episodes=sum(len(g) for g in groups.values()))


def main():
    state = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if state["stage"] != "confirmation_complete" or any(s["stage"] != "complete" for s in state["seeds"].values()):
        raise RuntimeError("Do not build final conclusions from incomplete runs")
    snapshot = json.loads((OUT / "source_snapshot.json").read_text(encoding="utf-8"))
    frozen = Path(snapshot["path"])
    for name, expected in snapshot["hashes"].items():
        if hashlib.sha256((frozen / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source mismatch: {name}")
    scenarios = json.loads((frozen / "configs/confirmation_v3.json").read_text(encoding="utf-8"))
    names = [s["name"] for s in scenarios]
    expected_pairs = {(name, seed) for name in names for seed in range(41000, 41005)}
    all_rows = []
    data, diagnostics = {}, {}
    for seed in range(3):
        folder = OUT / f"seed_{seed}"
        data[seed] = {}
        for kind in ["baseline", "teacher"]:
            checkpoint = torch.load(folder / kind / "latest.pt", map_location="cpu", weights_only=False)
            assert checkpoint["iteration"] == 7000 and checkpoint["total_steps"] == 1792000
            assert checkpoint["config"]["seed"] == seed and checkpoint["config"]["kind"] == kind
            assert checkpoint["config"]["env_profile"] == "robust_v2"
            log = rows(folder / kind / "train.jsonl")
            config_record = json.loads((folder / kind / "config.json").read_text(encoding="utf-8"))
            for name, expected in config_record['provenance']['file_sha256'].items():
                assert snapshot['hashes'].get(name) == expected, f'Training source mismatch: {name}'
            assert len(log) == (7000 if kind == "baseline" else 2000)
            assert log[-1]["curriculum_progress"] == 1
            if kind == "teacher":
                assert log[0]["curriculum_progress"] == 1
                assert checkpoint["config"]["curriculum_offset"] == 5000
        diagnostics[str(seed)] = {}
        for architecture in ["pooled_cnn", "ordered_mlp"]:
            checkpoint = torch.load(folder / architecture / "latest.pt", map_location="cpu", weights_only=False)
            assert checkpoint["iteration"] == 750 and checkpoint["config"]["architecture"] == architecture
            assert checkpoint["config"]["seed"] == seed and checkpoint["config"]["history"] == 50
            log = rows(folder / architecture / "train.jsonl")
            config_record = json.loads((folder / architecture / "config.json").read_text(encoding="utf-8"))
            for name, expected in config_record['provenance']['file_sha256'].items():
                assert snapshot['hashes'].get(name) == expected, f'Adaptation source mismatch: {name}'
            assert len(log) == 750
            diagnostics[str(seed)][architecture] = {
                key: float(np.mean([r[key] for r in log[-50:]]))
                for key in ["latent_mse", "target_latent_std", "velocity_error"]}
        for method in METHODS:
            evaluation = folder / ("eval_" + method)
            group = rows(evaluation / "episodes.jsonl")
            metadata = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
            assert len(group) == 55 and {(r["scenario"], r["eval_seed"]) for r in group} == expected_pairs
            assert metadata["arguments"]["horizon"] == 1000 and metadata["scenarios"] == scenarios
            assert metadata["arguments"]["history_mode"] == ("shuffled" if method == "ordered_shuffled" else "ordered")
            # Every training/evaluation process must refer to the frozen src/configs.
            for name, value in metadata["provenance"]["file_sha256"].items():
                assert snapshot["hashes"].get(name) == value, f"Execution source mismatch: {name}"
            for row in group:
                assert 0 < row["steps"] <= 1000
                assert row["tracking_success"] == (row["completed"] and row["velocity_rmse"] < .2 and row["yaw_rmse"] < .3)
                assert not (row["completed"] and row["fell"])
                assert all(np.isfinite(row[key]) for key in ["velocity_rmse", "yaw_rmse", "action_delta", "power"])
            data[seed][method] = group
            all_rows.extend(dict(training_seed=seed, method=method, **r) for r in group)
    assert len(all_rows) == 1155
    overall = {method: across_seeds({seed: data[seed][method] for seed in range(3)}) for method in METHODS}
    by_scenario = {name: {method: across_seeds({seed: [r for r in data[seed][method] if r["scenario"] == name] for seed in range(3)})
                          for method in METHODS} for name in names}
    paired = {}
    for comparison in ["baseline", "cnn", "constant", "zero", "ordered_shuffled", "oracle"]:
        differences = [overall["ordered"]["per_seed"][str(seed)]["tracking_success"] - overall[comparison]["per_seed"][str(seed)]["tracking_success"] for seed in range(3)]
        paired[comparison] = dict(per_seed=differences, mean=float(np.mean(differences)), std=float(np.std(differences, ddof=1)))
    recovery = {}
    for name in [s["name"] for s in scenarios if "change_step" in s]:
        recovery[name] = {}
        for method in METHODS:
            groups = [[r for r in data[seed][method] if r["scenario"] == name] for seed in range(3)]
            recovery[name][method] = [dict(recovered=sum(r["recovery_s"] is not None for r in group),
                                         mean_s=float(np.mean([r["recovery_s"] for r in group if r["recovery_s"] is not None])) if any(r["recovery_s"] is not None for r in group) else None,
                                         episodes=len(group)) for group in groups]
    result = dict(round=3, training_seeds=[0, 1, 2], episodes=1155, scenarios=scenarios,
                  labels=LABELS, methods=METHODS, overall=overall, by_scenario=by_scenario,
                  paired_ordered_success_difference=paired, recovery=recovery, adaptation_diagnostics=diagnostics,
                  budget=dict(actual_new_ppo_steps=6912000, actual_new_adaptation_steps=1152000,
                              actual_new_total_steps=8064000, baseline_branch_steps=1792000, rma_branch_steps=1984000))
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tables = OUT / "tables"
    tables.mkdir(exist_ok=True)
    with (tables / "all_episodes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    table_rows = []
    for name in names:
        for method in METHODS:
            for seed in range(3):
                group = [r for r in data[seed][method] if r["scenario"] == name]
                table_rows.append(dict(scenario=name, method=method, training_seed=seed, episodes=len(group), **seed_metrics(group)))
    with (tables / "per_seed_scenarios.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes, ["tracking_success", "fell"], ["Tracking success", "Fall rate"]):
        ax.bar(x, [overall[m]["mean"][metric] * 100 for m in METHODS],
               yerr=[overall[m]["std"][metric] * 100 for m in METHODS], capsize=4, color="#8bbecb")
        for seed in range(3):
            ax.scatter(x + (seed-1)*.1, [overall[m]["per_seed"][str(seed)][metric]*100 for m in METHODS], label=f"training seed {seed}", s=24)
        ax.set(ylim=(0, 120), ylabel="Percent", title=title)
        ax.axhline(100, color='grey', linewidth=.7, linestyle=':')
        ax.set_xticks(x, METHODS, rotation=25, ha="right")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("3 independent training seeds; 11 scenarios x 5 paired episodes per seed\nBars: seed mean; error bars: seed sample SD; dots: each seed")
    fig.tight_layout()
    fig.savefig(OUT / "overall.png", dpi=160)
    plt.close(fig)
    heat = np.array([[by_scenario[name][method]["mean"]["tracking_success"] for name in names] for method in METHODS])
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_yticks(range(len(METHODS)), METHODS)
    for y in range(len(METHODS)):
        for x in range(len(names)):
            ax.text(x, y, f"{heat[y,x]:.0%}", ha="center", va="center", color="white" if heat[y,x]>.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Tracking success (mean across 3 training seeds)")
    ax.set_title("All frozen scenarios, including failures; 15 episodes per cell")
    fig.tight_layout()
    fig.savefig(OUT / "scenario_success.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for seed in range(3):
        for kind in ["baseline", "teacher"]:
            log = rows(OUT / f"seed_{seed}" / kind / "train.jsonl")
            for ax, key in zip(axes.flat, ["velocity_error", "yaw_error", "falls", "action_delta"]):
                values = np.convolve([r[key] for r in log], np.ones(100)/100, mode="valid")
                ax.plot([r["total_steps"] for r in log[99:]], values, label=f"{kind} s{seed}", linewidth=1.1)
                ax.set(xlabel="Branch cumulative control steps", ylabel=key + " (100-update mean)")
                ax.grid(alpha=.2)
    axes[0,0].legend(ncol=2, fontsize=8)
    fig.suptitle("Teacher branches share their own baseline's first 1.28M steps")
    fig.tight_layout()
    fig.savefig(OUT / "training.png", dpi=160)
    plt.close(fig)

    base, ordered, constant = overall["baseline"], overall["ordered"], overall["constant"]
    delta = paired["baseline"]["mean"] * 100
    static_delta = paired["constant"]["mean"] * 100
    direction = "高于" if delta > 0 else "低于" if delta < 0 else "等于"
    text = ["# 三种子确认实验：四足运动自适应与时序消融", "",
            f"有序历史适应器的平均跟踪成功率为 {ordered['mean']['tracking_success']:.1%}，域随机化基线为 {base['mean']['tracking_success']:.1%}，前者{direction}后者 {abs(delta):.1f} 个百分点。固定隐变量为 {constant['mean']['tracking_success']:.1%}；有序适应与固定输入的平均差为 {static_delta:+.1f} 个百分点。这些是本轮固定终点、三个训练种子上的结果，不是原论文性能复现或真机验证。", "",
            "本轮完成 3 个独立初训种子、7 组策略、11 个场景、共 1,155 个配对评估回合。种子 0/1/2 分别从随机网络开始，教师与对应基线共享前 5,000 更新父策略；所有终点检查点在训练前固定。", "",
            "## 总体结果", "", "| 策略 | seed 0 成功 | seed 1 成功 | seed 2 成功 | 平均成功率 ± 种子SD | 跌倒/165 |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        item = overall[method]
        counts = [sum(r["tracking_success"] for r in data[seed][method]) for seed in range(3)]
        text.append(f"| {LABELS[method]} | {counts[0]}/55 | {counts[1]}/55 | {counts[2]}/55 | {item['mean']['tracking_success']:.1%} ± {item['std']['tracking_success']:.1%} | {item['falls']}/165 |")
    text += ["", "![三训练种子总体结果](../outputs/third_round/overall.png)", "", "误差条是三个训练种子均值的样本标准差；每个点代表一个完整训练种子。55 个回合共享同一策略，不能当作 55 次独立训练，也不做强显著性或普遍泛化声明。", "",
             "## 在线适应与时序对照", "", "| 有序历史成功率 − 对照 | seed 0 | seed 1 | seed 2 | 均值 ± 种子SD（百分点） |", "|---|---:|---:|---:|---:|"]
    for method, item in paired.items():
        text.append(f"| {LABELS[method]} | {item['per_seed'][0]*100:+.1f} | {item['per_seed'][1]*100:+.1f} | {item['per_seed'][2]*100:+.1f} | {item['mean']*100:+.1f} ± {item['std']*100:.1f} |")
    if static_delta <= 0 or delta <= 0:
        text += ["", "本轮未得到有序历史适应器同时优于固定隐变量和域随机化基线的总体结果。不能将个别场景改善或较小 latent MSE 写成整体自适应提升；工程亮点是实现闭环训练、独立种子对照、消融和失败诊断。"]
    else:
        text += ["", "本轮有序历史在总体均值上高于固定输入和域随机化基线，但仍需查看每个训练种子的差异与全场景结果。三种子的均值改善只能描述为当前仿真设置下的观察，不能推出真机效果或普遍鲁棒性。"]
    if all(value == 0 for value in paired['ordered_shuffled']['per_seed']):
        text += ['', '打乱历史没有改变任何训练种子的成功回合数，因此当前成功率指标未支持时间顺序必不可少的结论。有序MLP也未优于CNN。固定输入对照的收益集中在seed 0的低摩擦与摩擦突变场景；另两个种子没有同样的成功率改善。']
    text += ['', 'seed 1的基线与教师分支均未完成速度跟踪，部分回合能站满20秒但几乎不前进。它保留在所有聚合统计中，说明当前奖励/优化设置对初始化仍然敏感；本轮未单独操控这些因素，不能将失败原因确认为某一项超参数。']
    text += ["", "教师与基线累计 PPO 采样相同，但最后阶段教师学习率为 1e-4、基线保持 3e-4，且教师恢复时重置仿真/RNG；两者差异不能单独归因于特权信息。CNN、有序MLP、固定与零输入共享同一个终点教师，可用于判断该教师下适应器是否带来额外控制收益。", "",
             "CNN 与有序 MLP 都使用 H50（1 秒）历史，参数量相差小于 10%，但结构和优化路径不同，不能把差异只归因于时序信息。打乱历史在推理时改变输入分布，是顺序敏感性诊断。固定隐变量从训练参数分布抽取 4,096 样本后求教师编码均值，不使用测试场景真实参数；零隐变量可能偏离教师训练分布。", "",
             "## 全场景结果与边界", "", "![全场景跟踪成功率](../outputs/third_round/scenario_success.png)", "", "| 场景 | 基线成功/15 | CNN成功/15 | 有序MLP成功/15 | 固定输入成功/15 | 有序MLP跌倒/15 | 有序MLP速度RMSE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        items = by_scenario[name]
        text.append(f"| {name} | {items['baseline']['successes']}/15 | {items['cnn']['successes']}/15 | {items['ordered']['successes']}/15 | {items['constant']['successes']}/15 | {items['ordered']['falls']}/15 | {items['ordered']['mean']['velocity_rmse']:.3f} |")
    text += ["", "成功要求完成 20 秒、平面速度 RMSE < 0.2 m/s、偏航 RMSE < 0.3 rad/s；原地站立不会算作跟踪成功。RMSE 仅统计存活轨迹，须与跌倒、时长共同解读。只有 extrapolation 的参数超出训练范围；其他场景检验固定组合、参数突变和推扰。功率是控制步末端采样，不是电池能耗。", "",
             "参数突变恢复要求速度及偏航误差连续 0.5 秒达标，未恢复包含突变前跌倒。逐种子恢复数与已恢复回合的条件均值见 results.json；不能只报告已恢复回合的时间。推扰目前没有单独恢复时间定义。", "",
             "## 实现与预算", "", "- Unitree A1、12 关节，45 维本体观测、12 维动作，控制 50 Hz、物理 500 Hz。部署无根部线速度和真实动力学参数输入。", "- PPO 包含 GAE、tanh 动作概率修正、梯度裁剪；跌倒不 bootstrap，timeout 使用重置前状态 bootstrap，GAE 不跨回合。", "- 每种子 baseline 7,000 更新 / 1,792,000 步；teacher 从自身 baseline_5000 初始化，再 2,000 更新 / 512,000 新步，累计同为 1,792,000。教师最后阶段保持完整随机化范围，初始 actor/critic 与父策略精确相同。", "- 每种子两适应器各 750 更新 / 192,000 额外步；各 RMA 分支总预算 1,984,000，超过基线的总预算，不能称总采样预算相同。教师冻结，学生自身轨迹监督隐变量。", "- 本轮实际新采样 6,912,000 PPO 步 + 1,152,000 适应步 = 8,064,000 控制步，包含共享父策略节省的训练；每控制步为 10 物理步。", "- Windows 原生 MuJoCo 3.3.5、PyTorch 2.8.0+cpu；训练与仿真都在 CPU。硬件含 RTX 4060，但本项目没有 GPU 训练性能结论。", "", "![三种子训练曲线](../outputs/third_round/training.png)", "",
             "## 适应器诊断", "", "| 种子 | 适应器 | 最后50更新latent MSE | 目标latent标准差 | 在线训练速度误差 |", "|---|---|---:|---:|---:|"]
    for seed in range(3):
        for architecture, item in diagnostics[str(seed)].items():
            text.append(f"| {seed} | {architecture} | {item['latent_mse']:.6f} | {item['target_latent_std']:.6f} | {item['velocity_error']:.3f} |")
    text += ["", "latent MSE 与目标标准差来自学生训练轨迹；小 MSE 可能意味着容易预测或隐变量变化小，不代表控制成功。", "",
             "## 复跑与证据", "", "```powershell", ".venv/Scripts/python.exe tools/run_round3.py --out outputs/third_round_reproduction", ".venv/Scripts/python.exe -m rma_local.evaluate --checkpoint outputs/third_round/seed_0/ordered_mlp/latest.pt --mode rma --scenarios configs/confirmation_v3.json --seed 41000 --episodes 5 --out outputs/recheck_round3_ordered", "```", "",
             "第二条只重跑一个训练种子的有序策略。所有输出目录必须为新目录；不覆盖历史结果。完整训练复跑同预算，不保证逐位一致：续训恢复优化器，未恢复仿真/RNG状态。", "",
             "[训练前计划](round3_plan.md) / [全部回合CSV](../outputs/third_round/tables/all_episodes.csv) / [逐种子场景CSV](../outputs/third_round/tables/per_seed_scenarios.csv) / [结构化结果](../outputs/third_round/results.json) / [成果展示页](../portfolio/index.html)", "",
             "训练源码与场景冻结于 outputs/source_snapshots/round3_training；报告生成器检查预算、策略种类、种子、场景配对和每次评估的源码哈希。既有第一、二轮结果均保留。该项目是 RMA 核心方法的独立实现与工程变体，不含原论文复杂地形、全部环境因子、生物能耗奖励或实体机器人测试；传感器噪声与延迟尚未测试。", ""]
    latency_path = OUT / 'diagnostics/deployment_latency.json'
    if latency_path.exists():
        latency = json.loads(latency_path.read_text(encoding='utf-8'))
        text += ['', '## 单线程部署前向延迟', '', '| 模型 | 中位数(ms) | p95(ms) | p99(ms) | 演员及适应器参数量 |', '|---|---:|---:|---:|---:|']
        for method, item in latency['results'].items():
            text.append(f"| {LABELS[method]} | {item['median_ms']:.3f} | {item['p95_ms']:.3f} | {item['p99_ms']:.3f} | {item['parameters']:,} |")
        text += ['', '正式训练结束后独立测量：单线程、batch=1、预热200次、采样2000次，输入为零张量。包含Python调用和网络前向，不含传感器、历史更新或仿真；不能当作完整控制延迟或硬实时保证。并发评估中的延迟仅作原始记录，不用来宣称运行时优势。']
        result['deployment_latency'] = latency
    latent_paths = [OUT / f'diagnostics/latent_seed_{seed}.json' for seed in range(3)]
    if all(path.exists() for path in latent_paths):
        latent = [json.loads(path.read_text(encoding='utf-8')) for path in latent_paths]
        result['teacher_latent_sensitivity'] = latent
        text += ['', '## 教师对隐变量的动作敏感性', '', '| 训练种子 | 隐变量变化引起的动作标准差均值 | 与常量输入动作绝对差 | 常量与零输入动作绝对差 |', '|---|---:|---:|---:|']
        for seed, item in enumerate(latent):
            text.append(f"| {seed} | {item['action_std_mean']:.4f} | {item['action_deviation_from_constant_mean']:.4f} | {item['constant_vs_zero_action_mean_abs']:.4f} |")
        text += ['', '从训练参数范围固定抽取4096个样本；在名义教师轨迹观测上保持观测不变、改变隐变量，测量[-1,1]归一化动作变化。这用于区分编码器变化与演员是否利用隐变量，不是在线适应效果或跨状态泛化测试。']
    if (OUT / 'videos/nominal.mp4').exists():
        text += ['', '## 固定种子同步视频', '', '[名义运动](../outputs/third_round/videos/nominal.mp4) / [摩擦突变](../outputs/third_round/videos/friction_switch.mp4) / [范围外组合](../outputs/third_round/videos/extrapolation.mp4)。', '', '场景与展示种子在读取第三轮结果前固定；均为训练seed 0、评估seed 41000。左到右是域随机化基线、特权教师、CNN适应、有序MLP适应；提前结束明确标记，完整物理指标与原评估逐项一致。视频展示单回合，不能代表三种子整体成绩。']
    deployment_path = OUT / 'deployment/manifest.json'
    if deployment_path.exists():
        deployment = json.loads(deployment_path.read_text(encoding='utf-8'))
        result['deployment_export'] = deployment
        text += ['', '## 部署模型导出', '', '已将训练seed 0的基线、CNN及有序MLP导出为CPU TorchScript，只保留部署演员及适应器，移除评论家与特权编码器。随机batch=16、batch=1及历史冷启动输出与原模型在1e-6容差内一致。', '', '[部署清单与输入规范](../outputs/third_round/deployment/manifest.json) / [有序历史模型](../outputs/third_round/deployment/ordered.torchscript.pt) / [调用示例](demo_guide.md)。这是模型前向产物，尚未实现或验证真实机器人的传感器、关节映射、通信和运行安全。']
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "docs/third_results.md").write_text("\n".join(text), encoding="utf-8")
    portfolio = ROOT / "portfolio"
    portfolio.mkdir(exist_ok=True)
    view_data = dict(status="complete", training_seeds=[0, 1, 2], total_episodes=1155,
        headline=f"有序历史平均成功率 {ordered['mean']['tracking_success']:.1%}；域随机化基线 {base['mean']['tracking_success']:.1%}；固定隐变量 {constant['mean']['tracking_success']:.1%}。",
        interpretation=("当前结果未证明有序历史适应器同时优于基线和固定隐变量，报告保留所有失败与种子差异。" if static_delta <= 0 or delta <= 0 else "有序历史的总体均值高于基线和固定输入，但收益集中于seed 0；seed 1未学会跟踪，seed 2没有同样改善。CNN、有序MLP与打乱历史成功数相同，当前不支持时序模型更优或跨种子稳定提升。"),
        methods=[dict(method=method, label=LABELS[method],
                      success_by_seed=[sum(r['tracking_success'] for r in data[seed][method]) for seed in range(3)],
                      falls_by_seed=[sum(r['fell'] for r in data[seed][method]) for seed in range(3)],
                      episodes_per_seed=55, success_mean_pct=overall[method]['mean']['tracking_success']*100,
                      success_std_pct=overall[method]['std']['tracking_success']*100) for method in METHODS],
        scenario_rows=[dict(method=method, label=LABELS[method], scenario=name,
                            success_mean_pct=by_scenario[name][method]['mean']['tracking_success']*100,
                            velocity_rmse_mean=by_scenario[name][method]['mean']['velocity_rmse'],
                            falls=by_scenario[name][method]['falls'], episodes=15)
                       for name in names for method in METHODS],
        videos=[dict(label=name, src='../outputs/third_round/videos/'+name+'.mp4') for name in ['nominal','friction_switch','extrapolation'] if (OUT / 'videos' / (name+'.mp4')).exists()])
    (portfolio / 'data.js').write_text('window.ROUND3_DATA = '+json.dumps(view_data, ensure_ascii=False)+';\n', encoding='utf-8')
    print(json.dumps(dict(episodes=len(all_rows), overall={method: overall[method]["successes"] for method in METHODS}, paired=paired), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
