from __future__ import annotations
import numpy as np

class GaussianPolicy:
    """
    Linear Gaussian policy:
      action ~ N(W @ obs, sigma^2 I)
    """
    def __init__(self, obs_dim: int, act_dim: int, lr: float = 1e-2, sigma: float = 0.3, seed: int = 0):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

        # weights initialized small, reproducible
        self.W = self.rng.normal(0, 0.1, size=(act_dim, obs_dim))

    def act(self, obs: np.ndarray, rng: np.random.Generator):
        mean = self.W @ obs
        action = rng.normal(mean, self.sigma)
        return action, mean

    def update(self, trajectories: list[dict], baseline: float = 0.0):
        """
        trajectories: list of dicts with keys:
          obs, act, mean, ret
        baseline: scalar baseline to reduce variance
        """
        rets = np.array([step["ret"] for step in trajectories], dtype=np.float32)

        # normalize returns (variance reduction)
        if len(rets) > 1:
            rets = (rets - rets.mean()) / (rets.std() + 1e-8)

        grad_W = np.zeros_like(self.W)

        for step, G in zip(trajectories, rets):
            obs = step["obs"]
            act = step["act"]
            mean = step["mean"]

            advantage = G - baseline
            grad_logp = np.outer((act - mean) / (self.sigma ** 2), obs)
            grad_W += grad_logp * advantage

        grad_W /= len(trajectories)
        self.W += self.lr * grad_W
