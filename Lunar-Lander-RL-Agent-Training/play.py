import os

import gymnasium as gym
import imageio
import numpy as np
import torch

from src.network import QNetwork


def play():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array"
    )

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = QNetwork(
        state_size,
        action_size
    ).to(device)

    model.load_state_dict(
        torch.load(
            "models/lunarlander_ddqn.pth",
            map_location=device
        )
    )

    model.eval()

    os.makedirs("assets", exist_ok=True)

    rewards = []

    best_reward = -float("inf")
    best_frames = []

    episodes = 10

    print("=" * 70)
    print("Evaluating Trained Agent")
    print("=" * 70)

    for episode in range(episodes):

        state, _ = env.reset()

        done = False

        total_reward = 0

        frames = []

        while not done:

            frame = env.render()
            frames.append(frame)

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():

                q_values = model(state_tensor)

                action = torch.argmax(
                    q_values,
                    dim=1
                ).item()

            state, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            done = terminated or truncated
        final_frame = env.render()

        for _ in range(40):
            frames.append(final_frame)

        rewards.append(total_reward)

        print(
            f"Episode {episode + 1:2d}"
            f" | Reward: {total_reward:.1f}"
        )

        if total_reward > best_reward:

            best_reward = total_reward

            best_frames = frames.copy()

    env.close()

    imageio.mimsave(
        "assets/lunarlander_agent.gif",
        best_frames,
        fps=30,
        loop=0 
    )

    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)
    print(f"Average Reward : {np.mean(rewards):.2f}")
    print(f"Best Reward    : {np.max(rewards):.2f}")
    print(f"Worst Reward   : {np.min(rewards):.2f}")
    print("=" * 70)
    print("Best Gameplay GIF Saved:")
    print("assets/lunarlander_agent.gif")


if __name__ == "__main__":
    play()