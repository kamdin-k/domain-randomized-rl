import numpy as np
from src.envs.point_nav_env import EnvConfig, PointNavEnv

def main():
    cfg = EnvConfig(
        motion_noise_std=0.01,
        drift_std=0.01,
        obs_noise_std=0.005,
    )
    env = PointNavEnv(cfg, seed=0)
    obs = env.reset(seed=0)
    total = 0.0

    for _ in range(cfg.max_steps):
        action = np.random.uniform(-1, 1, size=(2,))
        obs, r, done, truncated, info = env.step(action)
        total += r
        if done or truncated:
            print("done:", done, "truncated:", truncated, "steps:", info["t"], "dist:", info["dist"])
            break

    print("total_reward:", round(total, 3), "obs_dim:", obs.shape)

if __name__ == "__main__":
    main()
