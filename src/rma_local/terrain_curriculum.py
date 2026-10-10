"""Distance-based terrain levels adapted from legged_gym (BSD-3-Clause).

Copyright (c) 2021 ETH Zurich, Nikita Rudin
Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
Adaptations copyright (c) 2026 local RMA project contributors.
See THIRD_PARTY_NOTICES.md and licenses/legged_gym.txt for license terms.

Source: legged_robot.py::_update_terrain_curriculum at commit
8fa29acc6fd1910c3d9659eef6310bdd301cde0a.
Adaptations: scalar/per-terrain levels, integrated changing commands,
forward corridor progress and survival guard, bounded final level.
"""
from dataclasses import dataclass, field


@dataclass
class TerrainCurriculum:
    max_level: int = 9
    levels: dict = field(default_factory=dict)

    def level(self, kind):
        return self.levels.get(kind, 0)

    def difficulty(self, kind):
        return 0.25 + 0.75 * self.level(kind) / self.max_level

    def update(self, kind, *, forward_progress, commanded_distance, failed, within_corridor, elapsed_s):
        if kind == "flat" or elapsed_s <= 0:
            return self.level(kind)
        # The upstream 8 m patch advances at half-length (4 m). Our patch
        # starts at 0.8 m and ends at 4 m, so 4.1 m means it was crossed.
        move_up = not failed and within_corridor and forward_progress >= 4.1
        # Integrate vx over the episode: the final command can be STOP.
        move_down = (failed or not within_corridor or
                     forward_progress < 0.5 * commanded_distance) and not move_up
        self.levels[kind] = max(0, min(self.max_level, self.level(kind) + int(move_up) - int(move_down)))
        return self.level(kind)
