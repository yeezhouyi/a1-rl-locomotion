# 复杂地形与多指令：开源选型与源码对应

核验日期：2026-10-10。实际读取固定提交的 README、配置与相关源码；不凭视频估计训练成本。

## 路线结论

现阶段保留 A1 / MuJoCo / PyTorch 的可解释实验链，移植 legged_gym 的**基于表现升降级的地形课程**。
现有盲行策略先完成同预算平地动态指令与地形训练对照；仅加台阶而保持旧训练逻辑不足以声称复现成熟地形方法。
GPU 训练候选优先检查 MuJoCo Playground 的 MJX/Warp 路线，采用独立 Linux 环境；其现成四足示例为 Go1，
不能把其预训练权重直接称作 A1 控制器。原项目的 RMA 动力学隐变量、历史适应器及多种子评估继续保留。

## 候选对照

| 项目 | 核验版本 | 值得借鉴的内容 | 本机与任务约束 | 本次使用状态 |
|---|---|---|---|---|
| [legged_gym](https://github.com/leggedrobotics/legged_gym/tree/8fa29acc6fd1910c3d9659eef6310bdd301cde0a) | `8fa29acc…` | A1 Rough 配置、按表现升降难度、指数跟踪奖励、周期指令 | README 使用 Python 3.6–3.8 / Torch 1.10 / Isaac Gym Preview 3 / rsl_rl v1.0.2；并已建议迁移 Isaac Lab | 改写地形课程规则；未宣称跑通 Isaac Gym |
| [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways/tree/0e7236bdc81ce855cbe3d70345a7899452bdeb1c) | `0e7236bd…` | Go1 步态参数条件化、Multiplicity of Behavior、指令课程 | README 推荐≥10GB显存，训练段描述默认约12GB；允许减少并行环境；8GB不能照搬默认配置 | 已读 README 与 gait 配置，作为后续多步态设计参考；尚未移植 |
| [rapid-locomotion-rl](https://github.com/Improbable-AI/rapid-locomotion-rl/tree/f5143ef940e934849c00284e34caf164d6ce7b6e) | `f5143ef9…` | Grid Adaptive Curriculum、基于 RMA 的 teacher/student | 提供 Go1 / Mini Cheetah；README Docker 路线为 Ubuntu18.04/20.04与 Isaac Gym Preview3，安装需要获取 NVIDIA 包 | 已读 README，作为历史适应与指令课程参考；尚未移植 |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/tree/d59156e099a8785dd58fbce2e25a6ee1f7b5800c) | `d59156e0…` | MJX/Warp GPU仿真、Go1 joystick/rough/handstand/getup、细分奖励 | pyproject要求Python≥3.11、MuJoCo/MJX≥3.6；README给出jax[cuda12]；Go1配置默认8192环境、2亿步，须缩小实测 | 已读 README/pyproject/joystick/PPO配置；GPU路线候选，未安装或训练 |

本机实测：RTX4060 Laptop，8188MiB；Windows驱动581.08；已有 Ubuntu24.04 WSL2、Python3.12.3，
WSL内 `nvidia-smi` 可识别同一显卡。**这只验证了设备可见，尚未验证 JAX、Warp 或 Isaac 训练。**
原 Windows 虚拟环境是 Torch2.8.0+cpu / MuJoCo3.3.5，不在原环境上混装GPU候选依赖。
WSL的虚拟磁盘空闲不能直接视为主机物理空闲，安装前还要按C盘实际剩余空间预算。

## 直接改写的规则与本地调整

原始源码：
[legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot.py#L421)。

上游：走过半个地形块升一级；不足 `指令平面速度 × 最大回合时长 × 0.5` 降一级；超过顶级随机返回较低等级。
本地 `terrain_curriculum.py`：每类地形等级0–9，难度 `0.25 + 0.75×level/9`；走过4.1m且回合存活、没有离开走廊才升阶；
跌倒、出走廊或不足累计前进指令距离的一半则降阶；等级封顶。保持级别不变也是合法结果。
改变指令时积分 `max(vx,0)×0.02`，防止终点恰好停止就丢掉整段运动要求。
仅自动训练回合结束后更新；固定评估地形不更新。每个并行环境保留各类地形等级，适应器新阶段从最低等级开始，
不宣称继承教师课程状态。课程状态也不随模型检查点恢复，和原本不恢复仿真状态的续训约定一致。

上游的地形块生成器依赖 Isaac Gym，无法原样放进 MuJoCo；本地坡/箱体台阶/正弦起伏仍是本地几何实现。
几何范围、类别比例、4秒指令重采样、10%停止概率是本地实验设置，均不是原仓库参数。
课程等级均值写入训练日志；没有走够就不强行升阶，若正式训练不能到高等级应如实报告。

## 不能照抄的配置

上游 [A1配置](https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/a1/a1_config.py)
使用位置目标PD，kp20、kd0.5、action_scale0.25；本项目kp30、kd0.8、action_scale0.5。
现有教师模型依赖旧动作语义，直接替换会改变控制器，必须单独从头训练才能公平比较。

上游基础粗糙地形配置 `num_observations=235`，包含基座线速度与187个高度扫描；本地45维观测不含这两者，
且特权编码器仅接收摩擦、载荷、强度。地形类型目前也不进入RMA隐变量。
因此课程移植只是**借鉴部分方法的A1盲行扩展**，不能称作完整复现 legged_gym 的 A1 Rough。

上游奖励有足部腾空时间、关节加速度、非足部碰撞、关节软限位；本地暂保留旧奖励，
便于分离“增加地形课程”与“修改奖励”的影响。下一轮如加奖励必须有单独同预算对照。
Walk these Ways 的多步态要额外提供步态参数/相位与奖励约束；目前的变速/转向/停止属于多指令控制，
不等于已学会 trot/pace/bound 等多步态，也没有技能切换网络或高层规划器。

## 许可证与可追溯证据

实际移植的 legged_gym 许可证全文与版权放在 `licenses/legged_gym.txt`、`THIRD_PARTY_NOTICES.md`。
Playground README声明 Apache2.0（rough纹理单列CC0）；Walk these Ways 与 Rapid Locomotion RL 顶层为MIT，
但都保留legged_gym/rsl_rl等分组件许可证，不能用一个MIT标签替代全部归属声明。
固定提交/所读文件哈希存放在 `configs/opensource_sources.json`，下载快照仅在忽略目录 `outputs/opensource_reference/`。

“4060训练高难度动作”的特定仓库仍未确定。现在能说8GB卡需要减少并行规模，不能承诺其训练时间或复现视频效果。

补充线索：WSL中已有用户Genesis fork `<GENESIS_REFERENCE>`，本地提交
`189e249b689dd8ccfc5bbc9ee28a7a15538dcd1f`。其 `examples/locomotion/go2_backflip.py`
是 Go2 单/双后空翻的**预训练策略回放**（1环境、torch.jit.load、没有训练循环），
`examples/locomotion/backflip/readme.md` 指向训练仓库
[ziyanx02/Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)。
本轮访问该训练仓库连接失败，未读其README/训练代码，不能确认4060要求。这个线索比单纯猜测Extreme Parkour更具体，
但仍不等于已识别用户记忆中的项目。Genesis本地README为Apache2.0、支持多平台；其4090速度数字是Franka仿真基准，
不能作为后空翻训练证据；该本地 `.venv` 尚无Torch，未运行演示或修改用户已有仓库。

## 简历价值与下一步验收

应展示：能阅读并迁移成熟方法、识别观测与控制器差异、设计等预算干预、定位失败并保留实验依据。
验收以本地完整回合的越障/跟踪/跌倒、各训练种子结果和可重跑视频为准；开源视频和仓库支持的技能不记作个人成果。
GPU候选应先通过独立环境的后端验证与小批量仿真吞吐/显存测量，再确定A1移植预算。

## 开发验证记录

`outputs/terrain_opensource_smoke` 完成单种子两次PPO更新/两次适应更新及7方法×12场景的30步短测；
它验证数据和调用链，不构成完整回合性能证据。
另以固定旧seed0教师运行256更新（65,536新控制步）开发试跑 `outputs/terrain_source_pilot_s0`，
只检查课程状态与数值稳定性，不使用正式确认集选择检查点。开发预算应与后续正式预算分开列示。
实测256更新完成，用时79.21秒、约827控制步/秒；上坡/下坡/起伏最高观察到level1，台阶level0。
采样期间共42次跌倒；这不是固定确认场景的跌倒率。第5轮正式330.24万步实验尚未启动。
17项测试通过，包括旧平地等价、地形高度、指令奖励时间对齐、课程边界及自动回合与固定评估的隔离。
这批开发证据不支持“已经学会复杂地形”，应先对照成熟足端奖励、观测与GPU采样路线再冻结正式方案。
