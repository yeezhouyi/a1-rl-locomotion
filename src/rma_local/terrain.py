"""Blind locomotion over physical slopes, vertical-riser stairs and rough ground."""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .env import A1Env
from .terrain_curriculum import TerrainCurriculum


class TerrainEnv(A1Env):
    def __init__(self, task_suite="terrain_v5", **kwargs):
        if task_suite not in ("terrain_v5", "commands_v5"):
            raise ValueError(task_suite)
        self.task_suite = task_suite
        self.terrain = dict(kind="flat")
        self.fixed_commands = False
        self.command_period = 200
        self.surface = np.zeros((81, 201))
        self.terrain_curriculum = TerrainCurriculum()
        self.automatic_terrain = False
        super().__init__(**kwargs)

    def build_world(self, root):
        root.find("worldbody").find("geom[@name='floor']").set("group", "5")
        ET.SubElement(root.find("asset"), "hfield", name="terrain", nrow="81", ncol="201", size="10 4 1 0.1")
        world = root.find("worldbody")
        ET.SubElement(world, "geom", name="terrain_surface", type="hfield", hfield="terrain", pos="8 0 0", material="ground_material", group="5", contype="0", conaffinity="0")
        for i in range(24):
            ET.SubElement(world, "geom", name=f"stair_{i}", type="box", pos="0 0 -5", size="0.2 4 0.01", material="ground_material", group="5", contype="0", conaffinity="0")

    def set_curriculum(self, progress):
        super().set_curriculum(progress)

    def sample_terrain(self):
        if self.task_suite == "commands_v5":
            return dict(kind="flat")
        kind = self.rng.choice(["flat", "uphill", "downhill", "stairs", "rough"], p=[.2, .2, .15, .3, .15])
        difficulty = self.terrain_curriculum.difficulty(str(kind))
        return dict(kind=str(kind), slope_degrees=float(self.rng.uniform(2, 8) * difficulty),
                    step_height=float(self.rng.uniform(.015, .045) * difficulty), step_width=.4,
                    amplitude=float(self.rng.uniform(.01, .03) * difficulty), phase=float(self.rng.uniform(0, 2*np.pi)))

    def configure_terrain(self, spec):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.terrain = dict(spec)
        kind = spec.get("kind", "flat")
        if kind not in ("flat", "uphill", "downhill", "stairs", "rough"):
            raise ValueError("Unknown terrain")
        self.surface.fill(0)
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        surface_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_surface")
        self.model.geom_contype[surface_id] = self.model.geom_conaffinity[surface_id] = int(kind in ("uphill", "downhill", "rough"))
        self.model.geom_rgba[surface_id, 3] = float(kind in ("uphill", "downhill", "rough"))
        self.model.geom_pos[self.floor_id, 2] = -1 if kind in ("uphill", "downhill", "rough") else 0
        x = np.linspace(-2, 18, 201)[None, :]
        y = np.linspace(-4, 4, 81)[:, None]
        if kind in ("uphill", "downhill"):
            degrees = float(spec.get("slope_degrees", 6))
            if not 0 < degrees <= 15:
                raise ValueError("Slope must be within (0, 15] degrees")
            rise = np.tan(np.deg2rad(degrees)) * np.clip(x - .8, 0, 3.2)
            self.surface[:] = rise if kind == "uphill" else np.tan(np.deg2rad(degrees)) * 3.2 - rise
        elif kind == "rough":
            amplitude = float(spec.get("amplitude", .02))
            if not 0 <= amplitude <= .08:
                raise ValueError("Roughness outside supported range")
            envelope = np.clip((x - .8) / .4, 0, 1)
            self.surface[:] = amplitude * envelope * (1 + .5*np.sin(x*9 + float(spec.get("phase", 0))) + .5*np.sin(y*8))
        self.model.hfield_data[:] = self.surface.ravel()
        for i in range(24):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"stair_{i}")
            enabled = kind == "stairs"
            self.model.geom_contype[gid] = self.model.geom_conaffinity[gid] = int(enabled)
            self.model.geom_pos[gid] = [0, 0, -5]
            if enabled:
                height = float(spec.get("step_height", .03))
                width = float(spec.get("step_width", .4))
                if not .0 < height <= .1 or not .2 <= width <= .8:
                    raise ValueError("Stair dimensions outside supported range")
                top = min(i + 1, 6) * height
                self.model.geom_pos[gid] = [.8 + (i + .5)*width, 0, top/2]
                self.model.geom_size[gid] = [width/2, 4, top/2]

    def ground_height(self, x, y):
        kind = self.terrain.get("kind", "flat")
        if kind == "flat":
            return 0.0
        if kind == "stairs":
            width = self.terrain.get("step_width", .4)
            i = int(np.floor((x - .8) / width))
            return min(i + 1, 6)*self.terrain.get("step_height", .03) if 0 <= i < 24 and abs(y) <= 4 else 0.0
        # Exact height via MuJoCo ray query, terrain group only (world geometry).
        point = np.array([x, y, 2.0])
        geomid = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data, point, np.array([0., 0., -1.]), np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8), 1, -1, geomid)
        return 2.0 - distance if distance >= 0 else -1.0

    def sample_command(self):
        if self.rng.random() < .1:
            return np.zeros(3, dtype=np.float32)
        return np.array([self.rng.uniform(.2, .75), self.rng.uniform(-.12, .12), self.rng.uniform(-.4, .4)], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = dict(options or {})
        # Only genuine automatic episode ends affect training difficulty.
        # Explicit fixed-scene resets and evaluation never train curriculum.
        if self.automatic_terrain and getattr(self, "episode_finished", False):
            self.terrain_curriculum.update(self.terrain["kind"],
                forward_progress=self.forward_progress, commanded_distance=self.commanded_distance,
                failed=self.episode_failed, within_corridor=self.within_corridor, elapsed_s=self.steps * .02)
        self.automatic_terrain = "terrain" not in options and self.task_suite == "terrain_v5"
        self.configure_terrain(options["terrain"] if "terrain" in options else self.sample_terrain())
        self.fixed_commands = bool(options.get("fixed_commands", "command" in options))
        obs, info = super().reset(seed=None, options=options)
        self.data.qpos[2] += self.ground_height(0., 0.)
        mujoco.mj_forward(self.model, self.data)
        if not self.fixed_commands:
            self.set_command(self.sample_command())
        self.commanded_distance = 0.0
        self.forward_progress = 0.0
        self.within_corridor = True
        self.episode_finished = self.episode_failed = False
        return self.observe(), info

    def step(self, action):
        self.commanded_distance += max(float(self.command[0]), 0.0) * .02
        obs, reward, term, timeout, info = super().step(action)
        self.forward_progress = max(self.forward_progress, float(self.data.qpos[0]))
        self.within_corridor &= abs(self.data.qpos[1]) < 3.8
        self.episode_finished = term or timeout
        self.episode_failed = term
        if not self.fixed_commands and self.steps % self.command_period == 0:
            self.set_command(self.sample_command())
            obs = self.observe()
        info["terrain"] = self.terrain["kind"]
        info["terrain_level"] = self.terrain_curriculum.level(self.terrain["kind"])
        info["terrain_difficulty"] = self.terrain_curriculum.difficulty(self.terrain["kind"])
        return obs, reward, term, timeout, info
