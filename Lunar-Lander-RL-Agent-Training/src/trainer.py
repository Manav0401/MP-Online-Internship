import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.agent import DDQNAgent


def train_ddqn():

    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    env = gym.make("LunarLander-v3")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DDQNAgent(state_size, action_size)
    episodes = 1000

    rewards = []

    best_reward = -float("inf")
    best_avg_reward = -float("inf")

    print("=" * 70)
    print("Training Double DQN on LunarLander-v3")
    print("=" * 70)

    progress_bar = tqdm(range(episodes))

    for episode in progress_bar:

        state, _ = env.reset()

        total_reward = 0
        done = False

        while not done:

            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            agent.train()

            state = next_state

            total_reward += reward
        agent.decay_epsilon()

        rewards.append(total_reward)

        best_reward = max(best_reward, total_reward)

        avg_reward = np.mean(rewards[-100:])

        if avg_reward > best_avg_reward:

            best_avg_reward = avg_reward

            torch.save(
                agent.policy_net.state_dict(),
                "models/lunarlander_ddqn.pth"
            )

        progress_bar.set_description(
            f"Ep {episode + 1:4d}"
            f" | Reward {total_reward:7.1f}"
            f" | Avg100 {avg_reward:7.1f}"
            f" | Best {best_reward:7.1f}"
            f" | ε={agent.epsilon:.3f}"
        )

        if len(rewards) >= 100 and avg_reward >= 200:

            print("\n" + "=" * 70)
            print("🎉 Environment Solved!")
            print(f"Average Reward (last 100 episodes) : {avg_reward:.2f}")

            torch.save(
                agent.policy_net.state_dict(),
                "models/lunarlander_ddqn.pth"
            )

            break

    torch.save(
        agent.policy_net.state_dict(),
        "models/lunarlander_ddqn_final.pth"
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        rewards,
        label="Episode Reward",
        alpha=0.4
    )

    if len(rewards) >= 100:

        moving_avg = np.convolve(
            rewards,
            np.ones(100) / 100,
            mode="valid"
        )

        plt.plot(
            range(99, len(rewards)),
            moving_avg,
            linewidth=3,
            label="100 Episode Average"
        )

    plt.axhline(
        y=200,
        color="green",
        linestyle="--",
        label="Solved Threshold (200)"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Double DQN - LunarLander-v3")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        "assets/lunarlander_rewards.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    env.close()

    print("\n" + "=" * 70)
    print("Training Complete")
    print(f"Best Episode Reward : {best_reward:.2f}")
    print(f"Best Average Reward : {best_avg_reward:.2f}")
    print("Best Model Saved  : models/lunarlander_ddqn.pth")
    print("Final Model Saved : models/lunarlander_ddqn_final.pth")
    print("=" * 70)