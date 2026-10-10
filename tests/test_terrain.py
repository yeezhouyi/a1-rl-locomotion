import mujoco
import numpy as np
from rma_local.terrain import TerrainEnv
from rma_local.env import A1Env
from rma_local.terrain_curriculum import TerrainCurriculum


def test_performance_curriculum_survival_and_integrated_commands():
    curriculum = TerrainCurriculum()
    good = dict(forward_progress=4.2, commanded_distance=6, failed=False, within_corridor=True, elapsed_s=20)
    assert curriculum.update("stairs", **good) == 1
    # Crossing before a later fall or by leaving the terrain cannot advance.
    assert curriculum.update("stairs", **dict(good, failed=True)) == 0
    assert curriculum.update("stairs", **dict(good, within_corridor=False)) == 0
    for _ in range(20):
        curriculum.update("stairs", **good)
    assert curriculum.level("stairs") == 9
    # STOP at the final instant must not erase earlier commanded distance.
    assert curriculum.update("stairs", **dict(good, forward_progress=1)) == 8
    assert curriculum.level("uphill") == 0
    assert curriculum.update("flat", **good) == 0
    assert curriculum.difficulty("stairs") <= 1


def test_curriculum_changes_only_after_finished_automatic_episode():
    env = TerrainEnv(randomize=False, horizon=1)
    try:
        env.reset(seed=9, options=dict(terrain=dict(kind="uphill", slope_degrees=1), command=[.4, 0, 0]))
        # Emulate the automatically sampled geometry without depending on RNG.
        env.automatic_terrain = True
        env.data.qpos[0] = 4.2
        env.data.qpos[2] = env.ground_height(4.2, 0) + .3
        mujoco.mj_forward(env.model, env.data)
        _, _, term, timeout, _ = env.step(np.zeros(12))
        assert timeout and not term
        assert env.terrain_curriculum.level("uphill") == 0
        env.reset()
        assert env.terrain_curriculum.level("uphill") == 1
        # Fixed evaluation geometry must not update the trained levels.
        for _ in range(2):
            env.reset(options=dict(terrain=dict(kind="uphill", slope_degrees=1), command=[.4, 0, 0]))
            env.step(np.zeros(12))
        assert env.terrain_curriculum.level("uphill") == 1
    finally:
        env.close()


def test_flat_terrain_extension_preserves_original_physics_and_observation():
    old, new = A1Env(randomize=False), TerrainEnv(randomize=False)
    try:
        options = dict(command=[.5, 0, 0], parameters=[.7, 0, 1])
        a, _ = old.reset(seed=8, options=options)
        b, _ = new.reset(seed=8, options=dict(**options, terrain=dict(kind="flat")))
        np.testing.assert_allclose(a, b)
        for _ in range(25):
            a, r1, t1, _, _ = old.step(np.zeros(12))
            b, r2, t2, _, _ = new.step(np.zeros(12))
            np.testing.assert_allclose(a, b, atol=1e-9)
            np.testing.assert_allclose(r1, r2, atol=1e-9)
            assert t1 == t2
    finally:
        old.close(); new.close()


def test_physical_terrain_heights_and_ground_relative_spawn():
    env = TerrainEnv(randomize=False)
    try:
        for spec in [dict(kind="uphill", slope_degrees=8), dict(kind="downhill", slope_degrees=8), dict(kind="stairs", step_height=.04, step_width=.4)]:
            env.reset(seed=5, options=dict(terrain=spec, command=[.5, 0, 0]))
            assert abs(env.data.qpos[2] - env.ground_height(0, 0) - .27) < .03
            if spec["kind"] == "uphill":
                assert abs(env.ground_height(2, 0) - np.tan(np.deg2rad(8))*1.2) < 1e-6
            if spec["kind"] == "stairs":
                assert env.ground_height(.99, 0) == .04
                assert env.ground_height(1.39, 0) == .08
                gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "stair_1")
                assert abs(env.model.geom_pos[gid, 2] + env.model.geom_size[gid, 2] - .08) < 1e-9
            for _ in range(10):
                obs, reward, _, _, info = env.step(np.zeros(12))
                assert np.isfinite(obs).all() and np.isfinite(reward) and "base_height" in info
    finally:
        env.close()


def test_command_update_is_visible_in_next_observation_and_not_previous_reward():
    env = TerrainEnv(task_suite="commands_v5", randomize=False)
    try:
        env.reset(seed=7, options=dict(command=[.5, 0, 0], fixed_commands=False))
        env.steps = 199
        previous_command = env.command.copy()
        obs, _, _, _, info = env.step(np.zeros(12))
        np.testing.assert_array_equal(info["command"], previous_command)
        np.testing.assert_array_equal(obs[6:9], env.command)
        assert not np.array_equal(env.command, previous_command)
    finally:
        env.close()
