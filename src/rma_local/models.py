import torch
from torch import nn
from torch.distributions import Normal

from .env import OBS_DIM, ACT_DIM


def mlp(dimensions):
    layers = []
    for i, (a, b) in enumerate(zip(dimensions, dimensions[1:])):
        layer = nn.Linear(a, b)
        nn.init.orthogonal_(layer.weight, gain=0.01 if i == len(dimensions) - 2 else 2 ** 0.5)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i < len(dimensions) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class Policy(nn.Module):
    def __init__(self, kind="teacher", latent_dim=8):
        super().__init__()
        self.kind = kind
        self.latent_dim = latent_dim
        self.encoder = mlp([3, 64, 64, latent_dim])
        self.actor = mlp([OBS_DIM + latent_dim, 128, 128, ACT_DIM])
        self.critic = mlp([OBS_DIM + latent_dim, 128, 128, 1])
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))

    def latent(self, obs, priv):
        if self.kind == "baseline":
            return torch.zeros((*obs.shape[:-1], self.latent_dim), device=obs.device)
        return self.encoder(priv)

    def features(self, obs, priv=None, latent=None):
        return torch.cat([obs, self.latent(obs, priv) if latent is None else latent], dim=-1)

    def value(self, obs, priv):
        return self.critic(self.features(obs, priv)).squeeze(-1)

    def distribution(self, obs, priv=None, latent=None):
        mean = self.actor(self.features(obs, priv, latent))
        return Normal(mean, self.log_std.clamp(-3., 1.).exp().expand_as(mean))

    def sample(self, obs, priv):
        distribution = self.distribution(obs, priv)
        raw = distribution.sample()
        action = torch.tanh(raw)
        log_prob = (distribution.log_prob(raw) - torch.log(1. - action.square() + 1e-6)).sum(-1)
        return action, raw, log_prob, self.value(obs, priv)

    def deterministic(self, obs, priv=None, latent=None):
        return torch.tanh(self.distribution(obs, priv, latent).mean)


class Adaptation(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.network = nn.Sequential(nn.Conv1d(OBS_DIM, 64, 5), nn.ELU(), nn.Conv1d(64, 64, 3), nn.ELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, latent_dim))

    def forward(self, history):
        return self.network(history.transpose(1, 2))


class OrderedAdaptation(nn.Module):
    """Preserve all H timestamps with approximately the H50 CNN's capacity."""
    def __init__(self, history=50, latent_dim=8):
        super().__init__()
        self.frame = mlp([OBS_DIM, 16])
        self.network = mlp([history * 16, 32, 32, latent_dim])

    def forward(self, history):
        return self.network(self.frame(history).flatten(start_dim=1))


def make_adaptation(architecture="pooled_cnn", history=50):
    if architecture == "pooled_cnn":
        return Adaptation()
    if architecture == "ordered_mlp":
        return OrderedAdaptation(history)
    raise ValueError(f"Unknown adaptation architecture: {architecture}")


def shift_history(history, obs, done=None):
    result = torch.cat([history[:, 1:], obs[:, None]], dim=1)
    if done is not None:
        result[done] = 0
        result[done, -1] = obs[done]
    return result
