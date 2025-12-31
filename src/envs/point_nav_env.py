from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class EnvConfig:
    dt: float = 0.1
    max_steps: int = 200
    goal_radius: float = 0.15
    world_size: float = 1.0  # positions in [-world_size, +world_size]
    action_scale: float = 0.2  # max delta per step (before dt)
    # domain randomization knobs
    motion_noise_std: float = 0.0
    drift_std: float = 0.0
    obs_noise_std: float = 0.0

class PointNavEnv:
    """
    Simple 2D point navigation.
    State (hidden): position (x,y)
    Observation: position + goal vector (dx, dy) with optional noise
    Action: (ax, ay) in [-1, 1] -> scaled movement
    Reward: -distance_to_goal each step, +1 bonus on success, -1 on timeout
    """
    def __init__(self, config: EnvConfig, seed: int | None = None):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.pos = self.rng.uniform(-0.8, 0.8, size=(2,))
        self.goal = self.rng.uniform(-0.8, 0.8, size=(2,))
        self.drift = self.rng.normal(0.0, self.cfg.drift_std, size=(2,))
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(2,)
        action = np.clip(action, -1.0, 1.0)

        # motion update with drift + noise
        delta = action * self.cfg.action_scale * self.cfg.dt
        noise = self.rng.normal(0.0, self.cfg.motion_noise_std, size=(2,))
        self.pos = self.pos + delta + self.drift * self.cfg.dt + noise

        # clip to world bounds
        self.pos = np.clip(self.pos, -self.cfg.world_size, self.cfg.world_size)

        self.t += 1

        dist = np.linalg.norm(self.goal - self.pos)
        done = dist <= self.cfg.goal_radius

        reward = -dist
        if done:
            reward += 1.0

        truncated = self.t >= self.cfg.max_steps
        if truncated and not done:
            reward -= 1.0

        info = {"dist": float(dist), "t": self.t}
        return self._obs(), float(reward), bool(done), bool(truncated), info

    def _obs(self):
        goal_vec = self.goal - self.pos
        obs = np.concatenate([self.pos, goal_vec], axis=0).astype(np.float32)
        if self.cfg.obs_noise_std > 0:
            obs += self.rng.normal(0.0, self.cfg.obs_noise_std, size=obs.shape)
        return obs
