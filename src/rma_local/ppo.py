import numpy as np
import math
import torch
from torch.nn import functional as F


def tanh_log_jacobian(raw):
    """Stable log(1-tanh(x)^2), including at saturated actions."""
    return 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))


def squashed_entropy(distribution, noise):
    """Reparameterized Monte Carlo entropy in normalized action coordinates.

    E[H(Normal) + log|J_tanh|]. External standard-normal noise lets the
    experiment isolate this sampling from rollout/minibatch RNG state.
    Differential entropy can be negative; this is not discrete entropy.
    """
    raw = distribution.mean + distribution.stddev * noise
    return distribution.entropy().sum(-1) + tanh_log_jacobian(raw).sum(-1)


def gae(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95):
    """Bootstrap time limits, but never propagate GAE across episode resets."""
    advantage = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
        advantage[t] = carry
    return advantage, advantage + values
