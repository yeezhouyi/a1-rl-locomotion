import numpy as np
import torch
import mujoco

from rma_local.env import A1Env, OBS_DIM
from rma_local.models import Adaptation, OrderedAdaptation, Policy, shift_history
from rma_local.ppo import gae


def test_time_limit_bootstraps_but_does_not_leak_next_episode():
    rewards = np.array([[1.], [100.]], dtype=np.float32)
    values = np.zeros((2, 1), dtype=np.float32)
    next_values = np.array([[2.], [0.]], dtype=np.float32)
    terminal = np.array([[False], [True]])
    timeout = np.array([[True], [False]])
    advantage, _ = gae(rewards, values, next_values, terminal, timeout, gamma=0.9)
    np.testing.assert_allclose(advantage[:, 0], [2.8, 100.])
    terminal[0] = True
    advantage, _ = gae(rewards, values, next_values, terminal, timeout, gamma=0.9)
    np.testing.assert_allclose(advantage[0], [1.])


def test_history_reset_erases_previous_episode():
    history = torch.ones(2, 10, OBS_DIM)
    obs = torch.full((2, OBS_DIM), 2.)
    result = shift_history(history, obs, torch.tensor([True, False]))
    assert not result[0, :-1].any()
    assert torch.all(result[:, -1] == 2.)
    assert torch.all(result[1, :-1] == 1.)


def test_baseline_and_student_do_not_need_privileged_parameters():
    obs = torch.randn(2, OBS_DIM)
    baseline = Policy("baseline")
    a = baseline.deterministic(obs)
    b = baseline.deterministic(obs, torch.randn(2, 3) * 100)
    torch.testing.assert_close(a, b)
    teacher = Policy("teacher")
    adaptation = Adaptation()
    predicted = adaptation(torch.randn(2, 50, OBS_DIM))
    action = teacher.deterministic(obs, latent=predicted)
    assert action.shape == (2, 12)
    assert action.abs().max() <= 1


def test_model_randomization_reaches_physics_without_entering_observation():
    env = A1Env(seed=1, randomize=False)
    try:
        env.reset(seed=2, options={"parameters": [0.7, 0., 1.], "command": [0.5, 0., 0.]})
        before = env.observe().copy()
        env.set_dynamics([0.25, 3., 0.7])
        np.testing.assert_allclose(before, env.observe(), atol=1e-6)
        assert abs(env.model.body_mass[1] - env.base_mass - 3.) < 1e-6
        assert np.allclose(env.model.geom_friction[:, 0], 0.25)
        for _ in range(20):
            obs, reward, term, timeout, _ = env.step(np.zeros(12))
            assert obs.shape == (OBS_DIM,)
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert abs(env.data.actuator_force).max() <= 33.5 * 0.7 + 1e-5
        # Forward recomputation legitimately changes solver warm-start values;
        # positions, velocities, time, actuations and external inputs must stay.
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        before_ctrl = env.data.ctrl.copy()
        before_force = env.data.xfrc_applied.copy()
        before_switch = np.empty(mujoco.mj_stateSize(env.model, spec))
        after_switch = np.empty_like(before_switch)
        mujoco.mj_getState(env.model, env.data, before_switch, spec)
        env.set_dynamics([0.9, 1., 1.1])
        mujoco.mj_getState(env.model, env.data, after_switch, spec)
        np.testing.assert_allclose(before_switch, after_switch, atol=1e-12)
        np.testing.assert_allclose(before_ctrl, env.data.ctrl)
        np.testing.assert_allclose(before_force, env.data.xfrc_applied)
        a, _ = env.reset(seed=20)
        env.step(np.ones(12))
        b, _ = env.reset(seed=20)
        np.testing.assert_allclose(a, b)
    finally:
        env.close()


def test_profile_changes_reward_without_changing_fixed_parameter_physics():
    legacy = A1Env(randomize=False, profile="legacy")
    robust = A1Env(randomize=False, profile="robust_v2")
    try:
        options = dict(parameters=[0.6, 1.1, 0.8], command=[0.4, 0., 0.])
        legacy.reset(seed=37, options=options)
        robust.reset(seed=37, options=options)
        for t in range(30):
            action = 0.1 * np.sin(np.arange(12) + t * 0.3)
            old_obs, old_reward, old_term, _, info = legacy.step(action)
            new_obs, new_reward, new_term, _, _ = robust.step(action)
            np.testing.assert_allclose(old_obs, new_obs, atol=1e-12)
            assert old_term == new_term
            expected = 0.5 * np.exp(-info["yaw_error"] ** 2 / 0.25) - 0.08 * info["action_delta"] - 3.0 * old_term
            np.testing.assert_allclose(new_reward - old_reward, expected, atol=1e-12)
    finally:
        legacy.close()
        robust.close()


def test_curriculum_only_changes_future_reset_distribution():
    env = A1Env(seed=19, profile="robust_v2", curriculum_progress=0.0)
    try:
        initial_parameters = env.parameters.copy()
        initial_observation = env.observe().copy()
        env.set_curriculum(1.0)
        np.testing.assert_array_equal(env.parameters, initial_parameters)
        np.testing.assert_array_equal(env.observe(), initial_observation)
        parameters = []
        for _ in range(32):
            env.reset()
            parameters.append(env.parameters.copy())
        parameters = np.asarray(parameters)
        assert np.all((parameters >= [0.25, 0., 0.65]) & (parameters <= [1., 3., 1.15]))
        assert parameters[:, 2].min() < 0.8
        assert parameters[:, 1].max() > 2.0
    finally:
        env.close()


def test_ordered_adaptation_uses_early_history_and_preserves_timestamp_order():
    torch.manual_seed(31415)
    model = OrderedAdaptation(history=50)
    history = torch.randn(2, 50, OBS_DIM, requires_grad=True)
    result = model(history)
    gradient = torch.autograd.grad(result.square().sum(), history)[0]
    assert result.shape == (2, 8)
    assert gradient[:, :5].abs().sum() > 0
    assert gradient[:, -5:].abs().sum() > 0
    assert not torch.allclose(result, model(history.flip(1)), atol=1e-7, rtol=1e-5)
    # Both candidates stay near 26k parameters; capacity is not identical.
    cnn_count = sum(p.numel() for p in Adaptation().parameters())
    ordered_count = sum(p.numel() for p in model.parameters())
    assert abs(cnn_count - ordered_count) / cnn_count < 0.1
