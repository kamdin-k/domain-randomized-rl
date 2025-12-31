from __future__ import annotations
import time
import numpy as np

from src.envs.point_nav_env import PointNavEnv, EnvConfig
from src.randomization.domain_randomizer import DomainRandomizer
from src.logging.csv_logger import CSVLogger

def main():
    out_path = f"results/rollout_{int(time.time())}.csv"

    base_cfg = EnvConfig(max_steps=200)
    dr = DomainRandomizer(seed=0)

    fieldnames = [
        "episode", "t",
        "pos_x", "pos_y", "goal_dx", "goal_dy",
        "act_x", "act_y",
        "reward", "done", "truncated", "dist",
        # domain params (metadata)
        "motion_noise_std", "drift_std", "obs_noise_std",
        "action_scale", "world_size", "goal_radius",
    ]

    with CSVLogger(out_path, fieldnames=fieldnames) as log:
        episodes = 3
        for ep in range(episodes):
            cfg = dr.sample_config(base_cfg)
            env = PointNavEnv(cfg, seed=ep)
            obs = env.reset(seed=ep)

            # keep same params for every step in this episode
            params = {
                "motion_noise_std": cfg.motion_noise_std,
                "drift_std": cfg.drift_std,
                "obs_noise_std": cfg.obs_noise_std,
                "action_scale": cfg.action_scale,
                "world_size": cfg.world_size,
                "goal_radius": cfg.goal_radius,
            }

            for _ in range(cfg.max_steps):
                action = np.random.uniform(-1, 1, size=(2,))
                next_obs, r, done, truncated, info = env.step(action)

                row = {
                    "episode": ep,
                    "t": info["t"],
                    "pos_x": float(obs[0]), "pos_y": float(obs[1]),
                    "goal_dx": float(obs[2]), "goal_dy": float(obs[3]),
                    "act_x": float(action[0]), "act_y": float(action[1]),
                    "reward": float(r),
                    "done": int(done),
                    "truncated": int(truncated),
                    "dist": float(info["dist"]),
                    **params,
                }
                log.log(row)

                obs = next_obs
                if done or truncated:
                    break

    print(f"✅ Wrote synthetic rollout data to: {out_path}")

if __name__ == "__main__":
    main()
