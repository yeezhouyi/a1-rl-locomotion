"""Write defensible resume and interview wording from final, audited results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/third_round"


def main():
    state = json.loads((OUT / "delivery_status.json").read_text(encoding="utf-8"))
    if state["stage"] != "ready_for_final_review":
        raise RuntimeError("Wait for full training, diagnostics and videos")
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    baseline = result["overall"]["baseline"]
    ordered = result["overall"]["ordered"]
    constant = result["overall"]["constant"]
    latency = result["deployment_latency"]["results"]["ordered"]["p95_ms"]
    samples = [round(ordered["per_seed"][str(seed)]["tracking_success"] * 55) for seed in range(3)]
    text = ["# 简历项目：RMA 四足机器人运动自适应", "",
            "建议项目名称：**基于 RMA 的四足机器人运动自适应：算法实现、扰动测试与消融实验**。", "",
            "## 可直接使用的三条", "",
            "- 基于 PyTorch、MuJoCo 与 Unitree A1，独立实现 PPO 特权教师、历史观测适应器及域随机化基线；采用 50 Hz 关节目标控制、500 Hz 仿射 PD，部署仅使用本体观测历史。",
            "- 构建 3 个独立训练种子、7 组策略、11 类固定场景的对照流程，完成 1,155 回合评估；比较 CNN 与有序历史 MLP，并以固定/零隐变量、打乱历史消融检验适应有效性，保存源码、配置、检查点与哈希。",
            f"- 实现约 5.27 万参数的有序历史部署模型，单线程 CPU 模型前向 p95 为 {latency:.3f} ms；导出仅含演员/适应器的 TorchScript 并核验动作一致性，结合同步回放、成功率与跌倒分析失效边界。", "",
            "第三条的延迟只含演员及适应器前向/Python调用，不含传感器、历史维护和仿真，不代表完整控制周期或硬实时保证。参数量包含部署演员和适应器，不包括不参与部署的教师编码器、评论家。", "",
            "## 一句话版", "",
            "基于 MuJoCo/PyTorch 独立实现 A1 四足 RMA 核心链路，完成三训练种子、7组策略和1,155回合扰动/消融评估，形成可复跑实验、同步视频与部署性能诊断。", "",
            "## 结果应怎样讲", "",
            f"第三轮有序历史的成功回合为 {samples[0]}/55、{samples[1]}/55、{samples[2]}/55，均值 {ordered['mean']['tracking_success']:.1%}；域随机化基线均值 {baseline['mean']['tracking_success']:.1%}，固定隐变量 {constant['mean']['tracking_success']:.1%}。误差与跌倒详情见完整报告，不能只挑一个最好种子。"]
    if ordered["mean"]["tracking_success"] <= baseline["mean"]["tracking_success"] or ordered["mean"]["tracking_success"] <= constant["mean"]["tracking_success"]:
        text += ["", "这轮没有证明有序历史在总体上同时优于基线和固定输入，因此简历突出自主实现、严格对照和部署工程，不写‘显著提升鲁棒性’。面试可讲为什么对照推翻了最初假设、如何排除‘只是站稳’和‘静态隐变量已够用’的解释。"]
    else:
        text += ["", "若要补充性能数字，只能写当前仿真固定集上的三种子均值差，并同时注明种子差异；三种子不足以做普遍泛化或真机改善声明。"]
    text += ["", "## 90 秒讲解", "",
             "我研究的问题是四足机器人在摩擦、载荷和执行器强度未知时，历史观测能否帮助速度跟踪。我用 MuJoCo 的 Unitree A1 模型，自己实现 PPO、特权编码器和控制器；训练教师时可读取动力学参数，部署时由历史适应器估计8维任务隐变量，再与当前观测一起输入演员。它预测的是任务相关表示，不直接回归物理参数。", "",
             "实现上我处理了两个容易出错的边界：跌倒不做价值自举，timeout使用重置前状态自举但GAE不跨回合；回合重置时清空历史，动力学突变时保留物理状态。适应器训练使用学生自身轨迹，减少教师与学生状态分布差异。", "",
             f"实验不是只拍一个视频。我从三个随机种子训练，固定7组对照、11个场景、1,155回合；有序历史平均成功率 {ordered['mean']['tracking_success']:.1%}，基线 {baseline['mean']['tracking_success']:.1%}，固定隐变量 {constant['mean']['tracking_success']:.1%}。我保留所有失败，用静态输入和打乱历史检查效果是否真来自在线适应。最后做了同步视频、可复跑证据和单线程部署前向测量。项目仍是平地CPU仿真，没有真机验证，也不是原论文全部实验的精确复现。", "",
             "## 岗位侧重", "",
             "- 机器人控制/强化学习岗位：重点讲部分可观测控制、PPO/GAE、特权训练、学生轨迹与动力学扰动。", "- 机器人算法工程岗位：重点讲仿真控制接口、参数突变状态保持、证据复跑、失败诊断和部署模型开销。", "- 研究类岗位：重点讲预先固定确认集、配对分叉、种子统计、额外采样预算和无法支持的因果归因。", "",
             "[第三轮报告](third_results.md) / [技术问答](interview_guide.md) / [展示入口](../portfolio/index.html) / [演示步骤](demo_guide.md)", ""]
    (ROOT / "docs/resume_project.md").write_text("\n".join(text), encoding="utf-8")
    guide = ROOT / "docs/interview_guide.md"
    content = guide.read_text(encoding="utf-8")
    content = content.replace("第三轮正式运行正在进行，当前不预先填写结果数字", "第三轮已完成三个训练种子与1,155回合固定确认评估，实际结果与最终90秒讲解见 [简历项目文案](resume_project.md)")
    content = content.replace("第三轮仍在正式运行，具体成功率、RMSE、跌倒数和提升百分比由最终日志回填。", "第三轮已完成，具体成功率、RMSE、跌倒数和预算见 [第三轮报告](third_results.md)；结论以实际对照为准。")
    if "## 主模型回填前核验点" in content:
        content = content.split("## 主模型回填前核验点")[0].rstrip() + "\n\n最终数字与证据入口见 [简历项目文案](resume_project.md) 与 [第三轮报告](third_results.md)。\n"
    guide.write_text(content, encoding="utf-8")
    print("Wrote docs/resume_project.md and updated interview guide.")


if __name__ == "__main__":
    main()
