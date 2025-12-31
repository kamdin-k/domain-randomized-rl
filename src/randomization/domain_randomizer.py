from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.envs.point_nav_env import EnvConfig

@dataclass
class RandRanges:
    # Each tuple is (min, max)
    motion_noise_std: tuple[float, float] = (0.0, 0.03)
    drift_std: tuple[float, float] = (0.0, 0.05)
    obs_noise_std: tuple[float, float] = (0.0, 0.02)
    action_scale: tuple[float, float] = (0.12, 0.30)
    world_size: tuple[float, float] = (0.8, 1.2)
    goal_radius: tuple[float, float] = (0.10, 0.20)

class DomainRandomizer:
    """
    Samples environment parameters each episode to test robustness.
    """
    def __init__(self, ranges: RandRanges | None = None, seed: int | None = None):
        self.ranges = ranges or RandRanges()
        self.rng = np.random.default_rng(seed)

    def sample_config(self, base: EnvConfig | None = None) -> EnvConfig:
        base = base or EnvConfig()
        r = self.ranges
        return EnvConfig(
            dt=base.dt,
            max_steps=base.max_steps,
            motion_noise_std=self._u(r.motion_noise_std),
            drift_std=self._u(r.drift_std),
            obs_noise_std=self._u(r.obs_noise_std),
            action_scale=self._u(r.action_scale),
            world_size=self._u(r.world_size),
            goal_radius=self._u(r.goal_radius),
        )

    def sample_params_dict(self) -> dict:
        cfg = self.sample_config()
        return {
            "motion_noise_std": cfg.motion_noise_std,
            "drift_std": cfg.drift_std,
            "obs_noise_std": cfg.obs_noise_std,
            "action_scale": cfg.action_scale,
            "world_size": cfg.world_size,
            "goal_radius": cfg.goal_radius,
        }

    def _u(self, a_b: tuple[float, float]) -> float:
        a, b = a_b
        return float(self.rng.uniform(a, b))
