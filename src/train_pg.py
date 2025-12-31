from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from src.envs.point_nav_env import PointNavEnv, EnvConfig
from src.randomization.domain_randomizer import DomainRandomizer
from src.agents.policy_gradient import GaussianPolicy

def run_episode(env, policy, rng):
    obs = env.reset()
    traj = []
    total_reward = 0.0

    for _ in range(env.cfg.max_steps):
        action, mean = policy.act(obs, rng)
        next_obs, r, done, truncated, info = env.step(action)
        traj.append({
            "obs": obs,
            "act": action,
            "mean": mean,
            "reward": r,
        })
        total_reward += r
        obs = next_obs
        if done or truncated:
            break

    # returns (Monte Carlo)
    G = 0.0
    for step in reversed(traj):
        G += step["reward"]
        step["ret"] = G

    return traj, total_reward

def moving_avg(x, w=20):
    if len(x) < w:
        return []
    return np.convolve(np.array(x), np.ones(w)/w, mode="valid")

def main():
    # global reproducibility
    rng = np.random.default_rng(0)

    base_cfg = EnvConfig(max_steps=200)
    dr = DomainRandomizer(seed=0)

    obs_dim = 4
    act_dim = 2
    policy = GaussianPolicy(obs_dim, act_dim, lr=8e-3, sigma=0.25, seed=0)

    rewards = []
    baseline = 0.0
    beta = 0.9  # EMA baseline smoothing

    episodes = 200
    for ep in range(episodes):
        cfg = dr.sample_config(base_cfg)
        env = PointNavEnv(cfg, seed=ep)

        traj, ep_reward = run_episode(env, policy, rng)

        # update baseline (EMA on episode reward)
        baseline = beta * baseline + (1 - beta) * ep_reward
        policy.update(traj, baseline=0.0)  # baseline already handled via normalization

        rewards.append(ep_reward)

        if ep % 20 == 0:
            print(f"ep {ep:03d} | reward {ep_reward:8.2f} | baseline {baseline:8.2f}")

    # plot raw + moving average
    plt.plot(rewards, label="episode reward")
    ma = moving_avg(rewards, w=20)
    if len(ma) > 0:
        plt.plot(range(19, 19 + len(ma)), ma, label="moving avg (20)")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Policy Gradient (Normalized Returns) + Domain Randomization")
    plt.legend()
    plt.savefig("results/train_curve.png", dpi=150)
    print("✅ Saved training curve to results/train_curve.png")

if __name__ == "__main__":
    main()
