# 第三方代码与设计来源

## legged_gym

仓库：https://github.com/leggedrobotics/legged_gym

固定版本：`8fa29acc6fd1910c3d9659eef6310bdd301cde0a`。

`src/rma_local/terrain_curriculum.py` 依据上游
`legged_gym/envs/base/legged_robot.py::_update_terrain_curriculum` 改写：
保留完成距离升阶、低于指令预期距离一半降阶的规则；改为每种地形单独保存等级，
积分动态前进指令，加入存活与走廊约束，最高等级封顶。

上游版权：Copyright (c) 2021 ETH Zurich, Nikita Rudin；
Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES。
原始许可证完整保存在 [licenses/legged_gym.txt](licenses/legged_gym.txt)。
该许可证是 BSD 三条款文本；资产与依赖有独立许可证，本项目没有导入上游机器人资产。

## MuJoCo Playground

仓库：https://github.com/google-deepmind/mujoco_playground

固定提交：`d59156e099a8785dd58fbce2e25a6ee1f7b5800c`。
GPU工具通过依赖继承 `go1/joystick.py` 的奖励、观测、指令及回合逻辑，
并参照 `go1/base.py` 初始化流程改写A1适配器。版权 Copyright 2026 Google LLC；
原始 Apache-2.0 许可证见 [licenses/mujoco_playground.txt](licenses/mujoco_playground.txt)。
粗糙地形直接采用该提交的 `go1/xmls/assets/hfield.png`（Apache-2.0），
来源与SHA256随生成的资产写入 `heightmap_source.json`。未导入 rocky_texture.png。
训练调用 Brax PPO；Brax 的依赖许可证随独立Python环境保留。
具体复用位置及与旧RMA的差异见 [GPU来源说明](docs/gpu_sources.md)。

GPU坡/台阶扩展的可选航向反馈参照固定legged_gym的
`legged_robot.py::_post_physics_step_callback`中heading误差生成yaw-rate指令的流程，
采用本地增益、限幅并加入通道横向误差；上述legged_gym版权与许可证同样适用。
本地台阶和坡道为MuJoCo box几何，并非复制上游地形生成器。

其他候选项目仅用于调研；未把它们的策略、奖励实现、模型权重或机器人资产导入本项目。
具体源码对应与复现边界见 [开源选型记录](docs/opensource_locomotion.md)。

## 已有机器人资产

本项目原有 Unitree A1 资源来自 MuJoCo Menagerie；下载版本及哈希记录在
`assets/unitree_a1/manifest.json`，资产目录保留其许可证。新地形课程不改变资产来源。
