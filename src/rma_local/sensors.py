"""Seeded sensor corruption without access to privileged state.

Angular velocity, gravity, joint position and scaled joint velocity are
delayed; commands and previous requested actions remain current.
"""
from collections import deque
import numpy as np
from .env import OBS_DIM


class SensorChannel:
    def __init__(self, seed, noise_scale=0.0, delay_steps=0):
        if noise_scale < 0 or delay_steps < 0 or int(delay_steps) != delay_steps:
            raise ValueError("Noise and integer sensor delay must be nonnegative")
        self.rng = np.random.default_rng(seed)
        self.noise_scale = noise_scale
        self.delay = int(delay_steps)
        self.queue = deque(maxlen=self.delay + 1)
        # Units: rad/s, dimensionless gravity, rad, and qdot*0.05.
        self.std = np.zeros(OBS_DIM, dtype=np.float32)
        self.std[:3] = 0.05
        self.std[3:6] = 0.01
        self.std[9:21] = 0.01
        self.std[21:33] = 0.025

    def apply(self, observation):
        current = np.asarray(observation, dtype=np.float32).copy()
        measured = current.copy()
        if self.noise_scale:
            measured += (self.rng.normal(size=OBS_DIM) * self.std * self.noise_scale).astype(np.float32)
            measured[3:6] /= max(float(np.linalg.norm(measured[3:6])), 1e-8)
        if not self.queue:
            self.queue.extend(measured.copy() for _ in range(self.delay + 1))
        else:
            self.queue.append(measured)
        result = self.queue[0].copy()
        result[6:9] = current[6:9]
        result[33:45] = current[33:45]
        return result
