"""Validate fourth-round evidence and generate tables, scientific figures and page."""
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fourth_round"
METHODS = ["baseline", "oracle", "ordered", "constant", "gaussian_teacher", "squashed_teacher"]
LABELS = dict(baseline="域随机化基线", oracle="原特权教师", ordered="原有序适应器", constant="原固定隐变量",
              gaussian_teacher="高斯熵续训教师", squashed_teacher="实际动作熵续训教师")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert read(OUT / "status.json")["stage"] == "experiment_complete"
    plan = read(OUT / "execution_plan.json")
    assert plan["new_ppo_steps"] == 1152000 and plan["eval_episodes"] == 432 and not plan["smoke"]
    snapshot = read(OUT / "source_snapshot.json")
    for name, expected in snapshot["hashes"].items():
        frozen = Path(snapshot["path"]) / name
        assert sha(frozen) == expected, name
        if name.startswith(("src/", "configs/")):
            assert sha(ROOT / name) == expected, name
    scenarios = read(ROOT / "configs/stress_v4.json")
    names = [s["name"] for s in scenarios]
    all_rows, per_seed = [], []
    for seed in range(3):
        folder = OUT / f"seed_{seed}"
        parents = read(folder / "parent_checkpoints.json")
        for item in parents.values():
            assert sha(Path(item["path"])) == item["sha256"]
        for mode in ["gaussian", "squashed"]:
            cp = torch.load(folder / f"{mode}_teacher/latest.pt", weights_only=False, map_location="cpu")
            config = cp["config"]
            assert cp["iteration"] == 7750 and cp["total_steps"] == 1984000
            assert config["entropy_mode"] == mode and config["iterations"] == 750
            assert config["learning_rate"] == 1e-4 and config["entropy_coefficient"] == 0.005
            assert config["curriculum_offset"] == 7000 and config["env_profile"] == "robust_v2"
            assert sha(Path(config["resume"])) == parents["oracle"]["sha256"]
        for method in METHODS:
            summary = read(folder / f"eval_{method}/summary.json")
            assert summary["scenarios"] == scenarios
            for name, expected in summary["provenance"]["file_sha256"].items():
                assert snapshot["hashes"][name] == expected, name
            source = folder / f"eval_{method}/episodes.jsonl"
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
            assert len(rows) == 24
            assert {(r["scenario"], r["eval_seed"]) for r in rows} == {(n, s) for n in names for s in range(61000, 61003)}
            for row in rows:
                assert row["tracking_success"] == (row["completed"] and row["velocity_rmse"] < 0.2 and row["yaw_rmse"] < 0.3)
                all_rows.append(dict(train_seed=seed, method=method, **row))
            for name in ["all", *names]:
                subset = rows if name == "all" else [r for r in rows if r["scenario"] == name]
                per_seed.append(dict(train_seed=seed, method=method, scenario=name, episodes=len(subset),
                    successes=sum(r["tracking_success"] for r in subset), falls=sum(r["fell"] for r in subset),
                    velocity_rmse=float(np.mean([r["velocity_rmse"] for r in subset])),
                    yaw_rmse=float(np.mean([r["yaw_rmse"] for r in subset])),
                    mean_steps=float(np.mean([r["steps"] for r in subset]))))
    assert len(all_rows) == 432
    overall = {}
    for method in METHODS:
        rows = [r for r in per_seed if r["method"] == method and r["scenario"] == "all"]
        rates = np.array([r["successes"] / r["episodes"] for r in rows])
        overall[method] = dict(successes_by_seed=[r["successes"] for r in rows], successes=sum(r["successes"] for r in rows),
            falls_by_seed=[r["falls"] for r in rows], falls=sum(r["falls"] for r in rows),
            mean_success=float(rates.mean()), seed_sd=float(rates.std(ddof=1)), velocity_rmse=float(np.mean([r["velocity_rmse"] for r in rows])))
    delta = (np.array(overall["squashed_teacher"]["successes_by_seed"]) - np.array(overall["gaussian_teacher"]["successes_by_seed"])) / 24 * 100
    sensor_degradation = []
    for seed in range(3):
        for method in METHODS:
            group = [r for r in all_rows if r["train_seed"] == seed and r["method"] == method]
            clean = {r["eval_seed"]: r for r in group if r["scenario"] == "clean"}
            for scenario in names[1:6]:
                stressed = [r for r in group if r["scenario"] == scenario]
                sensor_degradation.append(dict(train_seed=seed, method=method, scenario=scenario,
                    velocity_rmse_delta=float(np.mean([r["velocity_rmse"] - clean[r["eval_seed"]]["velocity_rmse"] for r in stressed])),
                    successes_delta=sum(r["tracking_success"] - clean[r["eval_seed"]]["tracking_success"] for r in stressed),
                    falls_delta=sum(r["fell"] - clean[r["eval_seed"]]["fell"] for r in stressed)))
    results = dict(episodes=432, seeds=[0, 1, 2], plan=plan, scenarios=scenarios, overall=overall,
                   paired_teacher_delta_pp=dict(by_seed=delta.tolist(), mean=float(delta.mean()), seed_sd=float(delta.std(ddof=1))), per_seed_scenarios=per_seed,
                   paired_sensor_degradation=sensor_degradation)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    training_diagnostics = []
    for seed in range(3):
        for mode in ["gaussian", "squashed"]:
            logs = [json.loads(l) for l in (OUT / f"seed_{seed}/{mode}_teacher/train.jsonl").read_text(encoding="utf-8").splitlines()]
            assert len(logs) == 750
            for window, selected in [("first50", logs[:50]), ("last50", logs[-50:])]:
                training_diagnostics.append(dict(train_seed=seed, entropy_mode=mode, window=window,
                    **{k: float(np.mean([r[k] for r in selected])) for k in ["action_entropy", "gaussian_entropy", "action_saturation", "velocity_error", "action_std", "gradient_norm", "clip_fraction"]}))
    tables = OUT / "tables"
    tables.mkdir(exist_ok=True)
    write_csv(tables / "all_episodes.csv", all_rows)
    write_csv(tables / "per_seed_scenarios.csv", per_seed)
    write_csv(tables / "paired_sensor_degradation.csv", sensor_degradation)
    write_csv(tables / "training_diagnostics.csv", training_diagnostics)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(METHODS):
        values = np.array(overall[method]["successes_by_seed"]) / 24 * 100
        ax.bar(i, values.mean(), color="#5790a7" if i < 4 else "#68a67c", alpha=0.7)
        ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="none", color="#34495e", capsize=4)
        ax.scatter(i + np.array([-.12, 0, .12]), values, c=["#ac473b", "#9269ad", "#22756b"], zorder=3)
    ax.set_xticks(range(6), ["Baseline", "Original\nteacher", "Original\nRMA", "Constant\nlatent", "Gaussian\ncontinued", "Squashed\ncontinued"])
    ax.set_ylabel("Tracking success (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Sensor stress: 3 training seeds; error bars = seed SD")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "overall.png", dpi=160); plt.close(fig)
    matrix = np.array([[sum(r["successes"] for r in per_seed if r["method"] == m and r["scenario"] == n) / 9 * 100 for n in names] for m in METHODS])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu")
    ax.set_xticks(range(8), names, rotation=25, ha="right")
    ax.set_yticks(range(6), METHODS)
    for i in range(6):
        for j in range(8):
            ax.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center", color="white" if matrix[i,j] > 65 else "#183147")
    fig.colorbar(im, ax=ax, label="Tracking success (%)")
    ax.set_title("9 paired episodes per cell (3 seeds x 3 reset seeds)")
    fig.tight_layout(); fig.savefig(OUT / "sensors.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in ["baseline", "ordered", "gaussian_teacher", "squashed_teacher"]:
        means, deviations = [], []
        for name in names[1:6]:
            values = [r["velocity_rmse_delta"] for r in sensor_degradation if r["method"] == method and r["scenario"] == name]
            means.append(np.mean(values)); deviations.append(np.std(values, ddof=1))
        ax.errorbar(np.arange(5), means, yerr=deviations, marker="o", capsize=3, label=method)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(range(5), names[1:6])
    ax.set_ylabel("Velocity RMSE change vs paired clean (m/s)")
    ax.set_title("Sensor-only degradation: mean and SD across 3 training seeds")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "sensor_degradation.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for seed in range(3):
        for mode in ["gaussian", "squashed"]:
            logs = [json.loads(l) for l in (OUT / f"seed_{seed}/{mode}_teacher/train.jsonl").read_text(encoding="utf-8").splitlines()]
            assert len(logs) == 750
            for col, metric in enumerate(["action_entropy", "velocity_error"]):
                data = np.array([r[metric] for r in logs])
                smoothed = np.convolve(data, np.ones(50) / 50, mode="valid")
                axes[seed,col].plot(np.arange(50,751), smoothed, label=mode)
                axes[seed,col].set_ylabel(f"seed {seed}: {metric}")
                axes[seed,col].grid(alpha=.2)
        axes[seed,0].legend()
    axes[-1,0].set_xlabel("Additional PPO updates"); axes[-1,1].set_xlabel("Additional PPO updates")
    fig.suptitle("50-update rolling means; training metrics are not test performance")
    fig.tight_layout(); fig.savefig(OUT / "entropy_training.png", dpi=160); plt.close(fig)
    lines = ["# 第四轮：实际动作熵对照与传感器压力测试", "", "本轮完成 115.2 万新增 PPO 步、三个父训练种子的同预算续训对照，以及 432 回合新场景测试。旧第三轮结果保留。", "",
             "## 总体结果", "", "|方法|seed 0成功/24|seed 1|seed 2|成功率均值 ± 种子SD|跌倒/72|", "|---|---:|---:|---:|---:|---:|"]
    table_rows = []
    for m in METHODS:
        v = overall[m]
        cells = [LABELS[m], *[str(n) for n in v["successes_by_seed"]], f'{v["mean_success"]*100:.1f}% ± {v["seed_sd"]*100:.1f}%', str(v["falls"])]
        lines.append("|" + "|".join(cells) + "|")
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    gaussian_falls = overall["gaussian_teacher"]["falls"]
    squashed_falls = overall["squashed_teacher"]["falls"]
    outcome = ("实际动作熵分支在该固定测试集的平均成功率更高，仍需检查种子差异。" if delta.mean() > 0 else
               "本次实际动作熵分支没有获得平均成功率提升；不把该改动推荐为已验证性能改进。")
    seed1_note = f'失败父种子seed 1的两条续训分支分别通过{overall["gaussian_teacher"]["successes_by_seed"][1]}/24和{overall["squashed_teacher"]["successes_by_seed"][1]}/24回合。'
    paired_falls = []
    for seed in range(3):
        for name in names:
            g = [r for r in all_rows if r["train_seed"] == seed and r["method"] == "gaussian_teacher" and r["scenario"] == name]
            s = [r for r in all_rows if r["train_seed"] == seed and r["method"] == "squashed_teacher" and r["scenario"] == name]
            ng, ns = sum(r["fell"] for r in g), sum(r["fell"] for r in s)
            if ng != ns:
                paired_falls.append(f"seed {seed} / {name}：高斯熵跌倒{ng}/3，实际动作熵跌倒{ns}/3")
    lines += ["", f"同预算的实际动作熵 − 高斯熵教师成功率差：逐种子 {delta.tolist()} 个百分点；均值 {delta.mean():+.1f}，种子SD {delta.std(ddof=1):.1f}。", "",
        f"{outcome} {seed1_note} 跌倒数分别为高斯熵{gaussian_falls}/72、实际动作熵{squashed_falls}/72。", "",
        "；".join(paired_falls) + "。跌倒减少也不自动等于速度跟踪达标。" if paired_falls else "两个续训分支没有逐场景跌倒数差异。", "",
        "两种新分支都是真实动力学可见的教师，不是新部署学生。与旧模型存在额外训练预算，只有两个续训分支能用于同预算熵目标对照。", "",
        "![总体与种子差异](../outputs/fourth_round/overall.png)", "", "## 逐场景结果", "",
        "|场景|baseline成功/9|原教师|原RMA|固定隐变量|高斯熵教师|实际动作熵教师|", "|---|---:|---:|---:|---:|---:|---:|"]
    scenario_html = []
    for name in names:
        numbers = [sum(r["successes"] for r in per_seed if r["method"] == m and r["scenario"] == name) for m in METHODS]
        lines.append("|" + "|".join([name, *map(str, numbers)]) + "|")
    for row in per_seed:
        if row["scenario"] != "all":
            cells = [LABELS[row["method"]], str(row["train_seed"]), row["scenario"], f'{row["successes"]}/3', str(row["falls"]), f'{row["velocity_rmse"]:.3f}', f'{row["mean_steps"]:.0f}']
            scenario_html.append(f'<tr data-method="{row["method"]}">' + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    lines += ["", "![传感器压力测试](../outputs/fourth_round/sensors.png)", "", "![配对传感器误差变化](../outputs/fourth_round/sensor_degradation.png)", "",
        "[逐种子配对退化指标](../outputs/fourth_round/tables/paired_sensor_degradation.csv)只比较共享物理设置的五个传感器干预场景与clean。负误差变化不等于适应能力增强；噪声可能改变控制轨迹，需结合成功与跌倒。", "", "## 实验机制与解释边界", "",
        "实际动作空间差分熵使用 H(Normal) + E[log|J_tanh|]，重参数化采样，log-Jacobian用softplus稳定计算；熵采样独立于控制轨迹随机流。负差分熵是允许的，不能把其绝对值当作探索成功率。", "",
        "噪声1x为角速度0.05rad/s、重力分量0.01（归一化）、关节位置0.01rad、关节速度0.5rad/s的独立高斯噪声。延迟20/40ms仅作用于传感器，命令与上一请求动作保持当前；冷启动复制初始观测。噪声强度是人为压力水平，无硬件实测依据。", "",
        "六个clean/噪声/延迟场景共享物理设置和指令，可用于观察传感器退化边界；另外两个场景同时改变动力学，不应只归因于传感器。相同训练模型的回合不是独立训练样本。速度RMSE仅在存活期间计算，须结合跌倒与步数。", "",
        "当前成功率门槛在部分场景产生地板效应：seed 1本来就不跟踪，40ms延迟也使各组成功数为0；因此同时保留连续误差、存活步数和跌倒，不能用相同成功数说明策略完全相同。", "",
        "![训练熵与速度误差](../outputs/fourth_round/entropy_training.png)", "", "[前后50更新优化诊断](../outputs/fourth_round/tables/training_diagnostics.csv)。训练轨迹分布会随策略变化，熵与误差的相关性不能代替控制测试。", "",
        "## 失败轨迹诊断", "", "冻结检查点名义指令0.5m/s时，seed 1基线平均实际vx=0.007m/s，教师约0m/s，均存活20秒。高斯熵与实际动作熵差异存在于成功及失败模型，不能据此直接认定失败根因。", "",
        "[最终逐步诊断](../outputs/fourth_diagnostics_final/summary.json) / [轨迹图](../outputs/fourth_diagnostics_final/trajectories.png)。早期 fourth_round_diagnostics 与 v2 使用近似截断Jacobian，不纳入本轮证据。", "",
        "## 复跑", "", "```powershell", ".venv/Scripts/python.exe tools/run_round4.py --out outputs/fourth_round_reproduction", ".venv/Scripts/python.exe -m rma_local.evaluate --checkpoint outputs/third_round/seed_0/ordered_mlp/latest.pt --mode rma --scenarios configs/stress_v4.json --seed 61000 --episodes 3 --out outputs/recheck_stress", "```", "",
        "续训恢复网络/Adam但不恢复仿真与RNG状态。保存源码快照、父模型SHA256、每次运行参数和全部失败；固定750更新终点，没有结果选检查点。只评估从已有模型出发的短程优化，未验证从头训练种子稳定性、地形、动作延迟、通信或真机。", "",
        "[预先固定计划](round4_plan.md) / [全部回合](../outputs/fourth_round/tables/all_episodes.csv) / [逐种子场景](../outputs/fourth_round/tables/per_seed_scenarios.csv) / [结构化结果](../outputs/fourth_round/results.json) / [进阶成果页](../portfolio/fourth.html)"]
    (ROOT / "docs/fourth_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    options = '<option value="all">全部方法</option>' + "".join(f'<option value="{m}">{LABELS[m]}</option>' for m in METHODS)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A1 RMA · 第四轮进阶实验</title>
<style>body{margin:0;background:#0b1020;color:#edf3ff;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1100px;margin:auto;padding:36px 20px}a{color:#68d5c3}h1{font-size:40px;line-height:1.25}h2{font-size:23px}.lead{font-size:18px;color:#bccae1}.panel{background:#121a2d;border:1px solid #26324b;border-radius:16px;padding:24px;margin:24px 0}.kpi{display:flex;gap:18px;flex-wrap:wrap}.kpi b{font-size:30px;color:#68d5c3}.kpi div{flex:1;min-width:160px;background:#182238;padding:18px;border-radius:12px}.note{color:#a4b3cb}table{width:100%;border-collapse:collapse;white-space:nowrap}td,th{padding:10px;border-bottom:1px solid #26324b;text-align:left}.scroll{overflow:auto}img{width:100%;border-radius:12px}select{background:#182238;color:#edf3ff;padding:9px;border:1px solid #60748b;border-radius:8px}summary{cursor:pointer}code{word-break:break-word}</style><body><main>
<a href="index.html">← 第三轮与对照视频</a><p style="color:#68d5c3">UNITREE A1 / EXPERIMENT 04</p><h1>从“能跑”走向<br>可诊断的运动控制</h1><p class="lead">检验实际动作熵的优化效果，并量化噪声与观测延迟下的失效边界。所有结果保留三个训练种子。</p>
<div class="kpi"><div><b>115.2 万</b><br>新增控制采样步</div><div><b>432 回合</b><br>固定新场景评估</div><div><b>6 个分支</b><br>三种子 × 两种熵目标</div></div>
<section class="panel"><h2>同预算干预：实际动作熵</h2><p>从相同父教师分叉，统一学习率、奖励、采样量与固定终点，仅改变熵目标。两个新分支均可读取特权参数；结果用于教师优化诊断。</p><div class="scroll"><table><thead><tr><th>方法</th><th>seed 0 /24</th><th>seed 1 /24</th><th>seed 2 /24</th><th>均值 ± 种子SD</th><th>跌倒 /72</th></tr></thead><tbody>OVERALL_ROWS</tbody></table></div><p class="note">同预算分支差：DELTA_TEXT。统计单位为训练种子，未从头验证训练稳定性。</p><img src="../outputs/fourth_round/overall.png" alt="三种子成功率与种子标准差"></section>
<section class="panel"><h2>传感器噪声与延迟</h2><p>1x：角速度 0.05 rad/s、重力分量 0.01、关节位置 0.01 rad、关节速度 0.5 rad/s。延迟为传感器通道20/40ms，命令及上一请求动作保持当前。人为压力设定，无真机数据。</p><img src="../outputs/fourth_round/sensors.png" alt="八场景传感器压力测试"><details><summary>展开逐种子场景结果</summary><label for="method-filter">方法筛选：</label><select id="method-filter">OPTIONS</select><div class="scroll"><table><thead><tr><th>方法</th><th>训练seed</th><th>场景</th><th>成功</th><th>跌倒</th><th>速度RMSE</th><th>平均步数</th></tr></thead><tbody>SCENARIO_ROWS</tbody></table></div></details></section>
<section class="panel"><h2>为什么做这个实验</h2><p>seed 1 在0.5 m/s名义指令下几乎不前进。高斯熵并不反映经过tanh后的动作熵，但成功模型也有饱和，不能凭诊断直接认定根因。本轮把假设变成同预算实验。</p><img src="../outputs/fourth_diagnostics_final/trajectories.png" alt="冻结模型速度、身高和动作饱和逐步诊断"><details><summary>查看续训曲线</summary><img src="../outputs/fourth_round/entropy_training.png" alt="动作熵与训练误差"></details><p class="note">当前平地CPU仿真；没有动作延迟、通信、复杂地形或真机。对照包含未改善与失败结果。MODEL_NOTE</p></section>
<p><a href="../docs/fourth_results.md">完整报告</a> · <a href="../docs/round4_plan.md">固定计划</a> · <a href="../outputs/fourth_round/tables/all_episodes.csv">全部432回合</a> · <a href="../docs/resume_project.md">简历文案</a></p></main><script>document.getElementById('method-filter').addEventListener('change',e=>{document.querySelectorAll('tr[data-method]').forEach(r=>r.hidden=e.target.value!=='all'&&r.dataset.method!==e.target.value)})</script></body></html>'''
    page = page.replace("OVERALL_ROWS", "".join(table_rows)).replace("SCENARIO_ROWS", "".join(scenario_html)).replace("OPTIONS", options)
    page = page.replace("DELTA_TEXT", f"逐种子 {', '.join(f'{d:+.1f}' for d in delta)} 百分点；均值 {delta.mean():+.1f} 百分点")
    page = page.replace("MODEL_NOTE", "新教师的结果不代表部署学生已经获得同样改进。")
    page = page.replace('<img src="../outputs/fourth_round/sensors.png" alt="八场景传感器压力测试">', '<img src="../outputs/fourth_round/sensors.png" alt="八场景传感器压力测试"><p style="color:#f4b860">本设置40ms观测延迟下，各方法均0/9跟踪成功；不是实机延迟阈值。</p><img src="../outputs/fourth_round/sensor_degradation.png" alt="配对观测干预的速度误差变化">')
    page = page.replace("<h2>同预算干预：实际动作熵</h2>", "<h2>同预算干预：实际动作熵</h2><p style=\"color:#f4b860\">本轮没有获得跟踪成功率提升，实际动作熵分支总跌倒更多；保留原方案。</p>")
    (ROOT / "portfolio/fourth.html").write_text(page + "\n", encoding="utf-8")
    print(json.dumps(dict(overall=overall, paired_delta=results["paired_teacher_delta_pp"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
