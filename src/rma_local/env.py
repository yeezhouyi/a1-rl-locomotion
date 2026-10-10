from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

from .settings import task_settings

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "assets" / "unitree_a1"
TRAIN_RANGES = np.array([[0.4, 1.0], [0.0, 2.0], [0.85, 1.15]])
OBS_DIM = 45
ACT_DIM = 12


def rotation(data):
    return data.xmat[1].reshape(3, 3)


class A1Env(gym.Env):
    """50 Hz position targets, 500 Hz torque-limited PD; MuJoCo Menagerie A1.

    Actor observations: local angular velocity, projected gravity, command,
    joint offsets/velocities, previous action. No base linear velocity or
    hidden dynamics parameters are returned in the actor observation.
    """

    def __init__(self, seed=0, randomize=True, horizon=1000, profile="legacy", curriculum_progress=1.0):
        if not (ASSET / "a1.xml").exists():
            raise FileNotFoundError("Run tools/fetch_assets.py first.")
        xml = ET.parse(ASSET / "a1.xml")
        root = xml.getroot()
        # MuJoCo's native Windows loader cannot reliably open Unicode paths.
        # Supply named resource bytes through its virtual filesystem instead.
        root.find("compiler").set("meshdir", "")
        root.find("compiler").set("texturedir", "")
        root.find("option").set("timestep", "0.002")
        ET.SubElement(root.find("asset"), "texture", name="ground_checker", type="2d", builtin="checker", rgb1="0.35 0.4 0.45", rgb2="0.55 0.6 0.65", width="256", height="256")
        ET.SubElement(root.find("asset"), "material", name="ground_material", texture="ground_checker", texrepeat="2 2", texuniform="true")
        root.find("worldbody").insert(0, ET.Element("geom", name="floor", type="plane", size="0 0 0.1", friction="0.8 0.02 0.01", material="ground_material"))
        self.build_world(root)
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.8 0.8 0.8")
        actuator = root.find("actuator")
        for item in list(actuator):
            actuator.remove(item)
            ET.SubElement(actuator, "general", name=item.attrib["name"], joint=item.attrib["joint"], gaintype="fixed", biastype="affine", gainprm="30", biasprm="0 -30 -0.8", forcelimited="true", forcerange="-33.5 33.5")
        resources = {p.name: p.read_bytes() for p in (ASSET / "assets").iterdir() if p.is_file()}
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=resources)
        self.data = mujoco.MjData(self.model)
        self.nominal = self.model.key_qpos[0, 7:].copy()
        self.base_mass = self.model.body_mass[1]
        self.base_inertia = self.model.body_inertia[1].copy()
        self.randomize = randomize
        self.settings = task_settings(profile)
        self.set_curriculum(curriculum_progress)
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (ACT_DIM,), dtype=np.float32)
        self.renderer = None
        self.reset(seed=seed)

    def set_curriculum(self, progress):
        """Only future resets use the new range; active episodes stay intact."""
        if not 0 <= progress <= 1:
            raise ValueError("Curriculum progress must be in [0, 1]")
        final = np.asarray(self.settings.train_ranges, dtype=float)
        self.train_ranges = TRAIN_RANGES + progress * (final - TRAIN_RANGES)

    def build_world(self, root):
        """Extension hook; default keeps historical flat tasks unchanged."""

    def ground_height(self, x, y):
        return 0.0

    def set_command(self, command):
        value = np.asarray(command, dtype=np.float32)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("Command must be finite [vx, vy, yaw_rate]")
        self.command = value.copy()

    def set_dynamics(self, parameters):
        friction, payload, strength = np.asarray(parameters, dtype=float)
        if friction <= 0 or payload < 0 or strength <= 0:
            raise ValueError("Invalid dynamics parameters")
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, state_spec))
        mujoco.mj_getState(self.model, self.data, state, state_spec)
        self.parameters = np.array([friction, payload, strength], dtype=np.float32)
        self.model.actuator_gainprm[:, 0] = 30. * strength
        self.model.actuator_biasprm[:, 1] = -30. * strength
        self.model.actuator_biasprm[:, 2] = -0.8 * strength
        self.model.actuator_forcerange[:] = [-33.5 * strength, 33.5 * strength]
        self.model.geom_friction[:, 0] = friction
        self.model.body_mass[1] = self.base_mass + payload
        # Payload is centered at the trunk COM; inertia is a simplified scaled model.
        self.model.body_inertia[1] = self.base_inertia * (self.base_mass + payload) / self.base_mass
        mujoco.mj_setConst(self.model, self.data)
        # mj_setConst uses qpos0 internally. Parameter switches must preserve
        # the actual integration state instead of teleporting the robot.
        mujoco.mj_setState(self.model, self.data, state, state_spec)
        mujoco.mj_forward(self.model, self.data)

    def privileged(self):
        # Fixed transforms, not statistics computed from the evaluation set.
        return (self.parameters - np.array([0.7, 1.0, 1.0], dtype=np.float32)) / np.array([0.3, 1.0, 0.15], dtype=np.float32)

    def observe(self):
        rot = rotation(self.data)
        return np.concatenate([
            self.data.qvel[3:6], rot.T @ np.array([0., 0., -1.]),
            self.command, self.data.qpos[7:] - self.nominal,
            self.data.qvel[6:] * 0.05, self.previous_action,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.previous_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.steps = 0
        self.command = np.asarray(options.get("command", [self.rng.uniform(0.2, 0.8), self.rng.uniform(-0.15, 0.15), self.rng.uniform(-0.4, 0.4)]), dtype=np.float32)
        params = options.get("parameters")
        if params is None:
            params = self.rng.uniform(self.train_ranges[:, 0], self.train_ranges[:, 1]) if self.randomize else [0.7, 0., 1.]
        self.set_dynamics(params)
        self.data.qpos[7:] += self.rng.uniform(-0.025, 0.025, ACT_DIM)
        mujoco.mj_forward(self.model, self.data)
        return self.observe(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1., 1.)
        target = np.clip(self.nominal + 0.5 * action, self.model.jnt_range[1:, 0], self.model.jnt_range[1:, 1])
        # Affine actuator realizes strength * (kp * (target-q) - kd * qdot)
        # with torque limits, entirely in MuJoCo; one native call for 10 steps.
        self.data.ctrl[:] = target
        mujoco.mj_step(self.model, self.data, nstep=10)
        energy = float(np.sum(np.abs(self.data.actuator_force * self.data.qvel[6:])))
        self.steps += 1
        rot = rotation(self.data)
        velocity = rot.T @ self.data.qvel[:3]
        # MuJoCo free-joint angular velocity is already in the local frame.
        angular = self.data.qvel[3:6]
        gravity = rot.T @ np.array([0., 0., -1.])
        velocity_error = float(np.linalg.norm(velocity[:2] - self.command[:2]))
        yaw_error = float(abs(angular[2] - self.command[2]))
        action_delta = float(np.mean((action - self.previous_action) ** 2))
        reward = (1.5 * np.exp(-velocity_error ** 2 / 0.25) + self.settings.yaw_weight * np.exp(-yaw_error ** 2 / 0.25)
                  + 0.2 - 0.5 * np.sum(gravity[:2] ** 2) - 0.2 * velocity[2] ** 2
                  - 0.01 * np.sum(angular[:2] ** 2) - self.settings.action_delta_weight * action_delta - 0.0002 * energy)
        local_height = self.data.qpos[2] - self.ground_height(*self.data.qpos[:2])
        terminated = bool(local_height < 0.16 or gravity[2] > -0.5 or not np.isfinite(self.data.qpos).all())
        if terminated:
            reward -= self.settings.fall_penalty
        self.previous_action = action.copy()
        info = dict(velocity_error=velocity_error, yaw_error=yaw_error, action_delta=action_delta, power=energy, forward_velocity=float(velocity[0]))
        info.update(base_height=float(local_height), world_x=float(self.data.qpos[0]), world_y=float(self.data.qpos[1]), command=self.command.tolist())
        return self.observe(), float(reward), terminated, self.steps >= self.horizon, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.qpos[:3]
        camera.distance, camera.azimuth, camera.elevation = 2., 135., -20.
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()


def make_env(task_suite="flat", **kwargs):
    if task_suite == "flat":
        return A1Env(**kwargs)
    from .terrain import TerrainEnv
    return TerrainEnv(task_suite=task_suite, **kwargs)


class BatchEnv:
    def __init__(self, count, seed=0, profile="legacy", curriculum_progress=1.0, task_suite="flat"):
        self.envs = [make_env(task_suite, seed=seed + i, profile=profile, curriculum_progress=curriculum_progress) for i in range(count)]
        self.obs = np.stack([e.observe() for e in self.envs])

    def set_curriculum(self, progress):
        for env in self.envs:
            env.set_curriculum(progress)

    def privileged(self):
        return np.stack([e.privileged() for e in self.envs])

    def step(self, actions):
        next_obs, rewards, terminals, timeouts, infos = zip(*(e.step(a) for e, a in zip(self.envs, actions)))
        final_obs = np.stack(next_obs)
        final_priv = self.privileged()
        self.obs = final_obs.copy()
        for i, (term, timeout) in enumerate(zip(terminals, timeouts)):
            if term or timeout:
                self.obs[i], _ = self.envs[i].reset()
        return self.obs.copy(), np.array(rewards), np.array(terminals), np.array(timeouts), final_obs, final_priv, infos

    def close(self):
        for env in self.envs:
            env.close()
