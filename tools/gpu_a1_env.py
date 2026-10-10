"""A1 port of MuJoCo Playground's Go1 joystick task.

Initialization adapted from go1/base.py at d59156e0 (Apache-2.0).
Copyright 2026 Google LLC. License: licenses/mujoco_playground.txt.
Reward, reset, observation and command code are inherited, not reimplemented.
This is a new 48D velocity-observed actor, NOT the original 45D RMA actor.
"""
from pathlib import Path
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import joystick


class A1Joystick(joystick.Joystick):
    def __init__(self, asset_dir, impl="warp", batch_size=128, noise=1.0, terrain="flat", reward_profile="upstream", reset_range=0.5):
        if terrain not in ("flat", "rough", "course"):
            raise ValueError(terrain)
        config = joystick.default_config()
        if not 0.5 <= reset_range <= 5.0:
            raise ValueError("reset_range must be between 0.5 and 5 metres")
        if terrain != "rough" and reset_range != 0.5:
            raise ValueError("Wide reset distribution is only defined for rough terrain")
        config.reset_xy_range = reset_range
        config.impl = impl
        config.Kp = 30.0
        config.Kd = 0.8
        config.noise_config.level = noise
        # Capacity for all robot/ground collisions, not only four feet.
        config.naconmax = 48 * batch_size
        config.njmax = 320
        config.command_config.a = [0.8, 0.2, 0.5]
        if reward_profile=="locomotion_v2":
            # Development hypothesis: reduce the stationary pose incentive and
            # increase the cost of ignoring commands. Keep upstream formulas.
            config.reward_config.scales.tracking_lin_vel=3.0
            config.reward_config.scales.tracking_ang_vel=1.0
            config.reward_config.tracking_sigma=.10
            config.reward_config.scales.pose=.1
            config.reward_config.scales.feet_clearance=-.5
            config.reward_config.scales.feet_air_time=.3
        elif reward_profile!="upstream":
            raise ValueError(reward_profile)
        self.reward_profile=reward_profile
        mjx_env.MjxEnv.__init__(self, config, None)
        asset_dir = Path(asset_dir)
        self._xml_path = str(asset_dir / f"a1_{terrain}.xml")
        self._model_assets = {p.name: p.read_bytes() for p in (asset_dir / "assets").iterdir() if p.is_file()}
        self._mj_model = mujoco.MjModel.from_xml_string(Path(self._xml_path).read_text(), assets=self._model_assets)
        self._mj_model.opt.timestep = config.sim_dt
        self._mj_model.opt.ccd_iterations = 20
        self._mj_model.dof_damping[6:] = config.Kd
        self._mj_model.actuator_gainprm[:, 0] = config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -config.Kp
        self._mj_model.vis.global_.offwidth = 640
        self._mj_model.vis.global_.offheight = 480
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=impl)
        self._imu_site_id = self._mj_model.site("imu").id
        self._feet_floor_found_sensor = [self._mj_model.sensor(f"{leg}_floor_found").id for leg in ["FR", "FL", "RR", "RL"]]
        self._post_init()

    def reset(self, rng):
        state = super().reset(rng)
        if self._config.reset_xy_range == 0.5:
            return state
        # Keep inherited yaw, velocities, command distribution and reset info.
        # Sample a wider region of the existing heightmap; no new terrain asset.
        xy = jax.random.uniform(jax.random.fold_in(rng, 20261010), (2,),
                                minval=-self._config.reset_xy_range,
                                maxval=self._config.reset_xy_range)
        data = state.data.replace(qpos=state.data.qpos.at[:2].set(xy))
        data = mjx.forward(self.mjx_model, data)
        return state.replace(data=data, obs=self._get_obs(data, state.info))

    def step(self, state, action):
        state = super().step(state, action)
        if self._config.reset_xy_range > 0.5:
            # Prevent training on the flat area beyond the finite heightmap.
            outside = jp.any(jp.abs(state.data.qpos[:2]) > 9.5)
            state = state.replace(done=jp.maximum(state.done, outside.astype(state.done.dtype)))
        return state
