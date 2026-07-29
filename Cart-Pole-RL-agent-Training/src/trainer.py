import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agent import DQNAgent


def train_dqn():

    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    episodes = 500

    rewards = []

    best_reward = 0
    best_avg_reward = 0

    print("=" * 70)
    print("Training Dueling Double DQN on CartPole-v1")
    print("=" * 70)

    for episode in range(episodes):

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

        rewards.append(total_reward)

        best_reward = max(best_reward, total_reward)

        avg_reward = np.mean(rewards[-20:])

        if avg_reward > best_avg_reward:

            best_avg_reward = avg_reward

            torch.save(
                agent.policy_net.state_dict(),
                "models/dqn_cartpole_best.pth"
            )

            print(
                f"\n🏆 New Best Average Reward : {best_avg_reward:.2f}"
            )

        print(
            f"Episode {episode + 1:3d}"
            f" | Reward {total_reward:3.0f}"
            f" | Avg20 {avg_reward:6.2f}"
            f" | Best {best_reward:3.0f}"
            f" | ε={agent.epsilon:.3f}"
        )


        if len(rewards) >= 20 and avg_reward >= 475:

            print("\n" + "=" * 70)
            print("🎉 Environment Solved!")
            print(f"Average Reward : {avg_reward:.2f}")

            torch.save(
                agent.policy_net.state_dict(),
                "models/dqn_cartpole_best.pth"
            )

            break

    torch.save(
        agent.policy_net.state_dict(),
        "models/dqn_cartpole_final.pth"
    )


    plt.figure(figsize=(12, 6))

    plt.plot(
        rewards,
        label="Episode Reward",
        alpha=0.4
    )

    if len(rewards) >= 20:

        moving_avg = np.convolve(
            rewards,
            np.ones(20) / 20,
            mode="valid"
        )

        plt.plot(
            range(19, len(rewards)),
            moving_avg,
            linewidth=3,
            label="20 Episode Average"
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Dueling Double DQN - CartPole-v1")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        "assets/reward_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    env.close()

    print("\n" + "=" * 70)
    print("Training Complete")
    print(f"Best Episode Reward : {best_reward}")
    print(f"Best Average Reward : {best_avg_reward:.2f}")
    print("Best Model Saved : models/dqn_cartpole_best.pth")
    print("Final Model Saved : models/dqn_cartpole_final.pth")
    print("=" * 70)