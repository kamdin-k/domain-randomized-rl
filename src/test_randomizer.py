from src.randomization.domain_randomizer import DomainRandomizer
from src.envs.point_nav_env import EnvConfig

def main():
    base = EnvConfig()
    dr = DomainRandomizer(seed=123)

    for i in range(5):
        cfg = dr.sample_config(base)
        print(f"[{i}] motion_noise={cfg.motion_noise_std:.4f} drift={cfg.drift_std:.4f} "
              f"obs_noise={cfg.obs_noise_std:.4f} action_scale={cfg.action_scale:.3f} "
              f"world={cfg.world_size:.2f} goal_r={cfg.goal_radius:.3f}")

if __name__ == "__main__":
    main()
