# GPU A1 实验的来源与边界

## 来源与固定版本

本实验使用本仓库 `outputs/opensource_reference/mujoco_playground` 中的 MuJoCo Playground 快照，固定 HEAD：`d59156e099a8785dd58fbce2e25a6ee1f7b5800c`。对应上游许可证为 Apache-2.0（仓库 `LICENSE`）。本地任务适配代码在 `tools/gpu_a1_env.py`，训练入口在 `tools/train_gpu_a1.py`。
许可证全文保存于 `licenses/mujoco_playground.txt`。依赖、编译缓存和训练结果位于独立目录 `<GPU_WORKSPACE>`；该目录挂载自 D 盘 `<GPU_STORAGE>/training.ext4` 的 20 GiB ext4 镜像，不写入原有 WSL 项目环境。
Windows 日志及导出结果位于 `<EVIDENCE>`。启动器通过 `launch_gpu_task.sh` 在同一WSL调用中重新挂载、以普通用户运行、导出结果；没有修改系统 fstab。
启动器要求 C 盘至少剩余25 GiB、D盘至少15 GiB、训练镜像内部至少3 GiB，且目录必须已经挂载。清理WSL内部的本任务缓存不会自动缩小C盘上的原WSL虚拟盘，因此不把缓存删除量描述为C盘立即恢复的空间。

## 实际复用与改写

`A1Joystick` 继承 Playground 的 `joystick.Joystick`。reset、step、observation、reward、指令采样等任务逻辑由 Joystick 实现继承而来；A1 适配没有重新实现这些算法。初始化则参照该快照 `go1/base.py` 的 `Go1Env` 初始化流程改写：读取本地 A1 XML/资源，设置 MuJoCo/MJX 模型、仿真步长、PD 参数和碰撞容量。模型由 Go1 改为 A1，且适配代码显式调整了若干配置，因此并非原样运行 Go1 环境。

训练器使用上游 Brax PPO（`brax.training.agents.ppo.train` 与 `make_ppo_networks`），由 Playground 的 `wrapper.wrap_for_brax_training` 包装环境；训练参数与网络宽度在 `tools/train_gpu_a1.py` 定义。A1 XML、网格、纹理来源由 `assets/unitree_a1/manifest.json` 逐项记录，来源是 MuJoCo Menagerie 的 `unitree_a1` 文件；清单同时记录文件 SHA-256。
粗糙地形模型直接复用同提交 `go1/xmls/assets/hfield.png`，沿用20m×20m占地及0.05m高度缩放；
图像像素决定实际局部高度，并非处处为5cm障碍。生成器保存哈希与来源。未复制上游Go1机器人网格或外部岩石纹理。
A1保留机身和腿部地面碰撞，禁止自碰撞，区别于上游默认feet-only模型；关节力矩上限仍为33.5Nm。

## 观测、奖励与对比边界

上游 Joystick 的 `state` actor 观测是 48 维：局部线速度、陀螺仪、投影重力、关节角偏差、关节速度、前一动作和速度指令。它包含线速度；本仓库旧环境的 45 维 actor 观测明确不含基座线速度。因此这是新任务/新策略设定，不是把旧策略原封不动搬到 GPU。

训练网络将 `state` 用于 policy，将包含额外传感器、接触、足端速度等信息的 `privileged_state` 用于 value/critic。critic 的特权观测不等于 RMA：这条训练配置没有 RMA 的历史编码器、潜变量适应模块及对应训练/部署流程。

Joystick 自带的奖励、动作映射、命令采样和终止条件，与旧 `src/rma_local/env.py` 的奖励及控制实现不同。故两者跨环境框架、任务、奖励和策略观测直接比较不构成公平的等条件基准；本实验也不能称作完整论文复现。结果应描述为“基于 MuJoCo Playground Joystick 任务逻辑、改写为 A1 模型并使用 Brax PPO 的 GPU 实验”。

本文记录代码来源和方法边界，不报告或暗示尚未完成的 GPU 测试结果。

追加的可选宽起点训练覆盖±3m（上游±0.5m）：继承reset后替换xy并重新forward及生成obs，保留随机yaw、初速、指令和info；step继承后将地图越界作为终止。默认±0.5m维持原任务行为。该扩展基于新空间起点失败诊断，不是上游原样实现。

后续坡地/台阶路线为`gpu_course_env.py`：继承原A1环境，加入本地box几何、表面高度函数、地形起点/命令分布、相对足端净空和越界终止；关闭绝对z摆足峰值奖励。`evaluate_gpu_course.py`的可选航向反馈参照固定legged_gym`legged_robot.py::_post_physics_step_callback`，新增本地横向误差项和增益，不照抄原任务参数。该高层使用仿真yaw/xy位置，不能宣称全部信息来自48维底层策略。视频和评估明确标注启用反馈；几何核验、结果和任务预算见`gpu_course_results.md`。

## 实测发现的依赖兼容性

- 自动解析的JAX0.11.2已移除`device_put_replicated`，Brax0.14.2训练初始化仍调用它，实跑失败。因此训练固定JAX/CUDA插件到0.8.2。
- Playground的`mjx_env.put_model`显式import `mujoco_warp`，自动依赖未安装该独立包；显式加入`mujoco-warp==3.15.0`后物理试跑成功。
- Brax0.14.2保存`mean_kernel_init_fn=null`，但`load_config`把None当字典key查找，独立恢复触发`KeyError: None`。本地`gpu_checkpoint.py`保留None语义，读取保存的网络配置并调用上游`checkpoint.load`恢复参数和normalizer；不修改原检查点或第三方安装文件。
- 最终实际依赖版本以D盘`requirements-frozen.txt`和每次实验`config.json`为准，原失败/旧版本探针不删，避免混淆版本。
