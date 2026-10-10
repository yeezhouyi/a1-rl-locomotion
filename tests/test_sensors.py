import numpy as np
import pytest
from rma_local.sensors import SensorChannel


def test_clean_sensor_channel_is_identity():
    obs = np.arange(45, dtype=np.float32)
    channel = SensorChannel(4)
    np.testing.assert_array_equal(channel.apply(obs), obs)
    np.testing.assert_array_equal(channel.apply(obs + 1), obs + 1)


def test_delay_preserves_current_commands_and_previous_actions():
    channel = SensorChannel(1, delay_steps=2)
    initial = np.arange(45, dtype=np.float32)
    channel.apply(initial)
    one = channel.apply(initial + 100)
    two = channel.apply(initial + 200)
    three = channel.apply(initial + 300)
    np.testing.assert_array_equal(one[:6], initial[:6])
    np.testing.assert_array_equal(two[9:33], initial[9:33])
    np.testing.assert_array_equal(three[:6], initial[:6] + 100)
    np.testing.assert_array_equal(three[6:9], initial[6:9] + 300)
    np.testing.assert_array_equal(three[33:], initial[33:] + 300)


def test_seeded_noise_is_reproducible_and_has_no_command_noise():
    obs = np.zeros(45, dtype=np.float32)
    obs[5] = -1
    a, b = SensorChannel(2, noise_scale=1), SensorChannel(2, noise_scale=1)
    actual = a.apply(obs)
    np.testing.assert_array_equal(actual, b.apply(obs))
    assert not np.array_equal(actual[:3], obs[:3])
    np.testing.assert_array_equal(actual[6:9], obs[6:9])
    np.testing.assert_array_equal(actual[33:], obs[33:])
    assert abs(np.linalg.norm(actual[3:6]) - 1) < 1e-6
    with pytest.raises(ValueError):
        SensorChannel(2, delay_steps=0.5)
