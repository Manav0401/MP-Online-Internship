import os

import gymnasium as gym
import imageio
import numpy as np
import torch

from src.network import DQN


def play():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = gym.make(
        "CartPole-v1",
        render_mode="rgb_array"
    )

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = DQN(
        state_size,
        action_size
    ).to(device)

    model.load_state_dict(
        torch.load(
            "models/dqn_cartpole_best.pth",
            map_location=device
        )
    )

    model.eval()

    os.makedirs("assets", exist_ok=True)

    rewards = []

    best_reward = -1
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

        # Hold last frame
        final_frame = env.render()

        for _ in range(15):
            frames.append(final_frame)

        rewards.append(total_reward)

        print(
            f"Episode {episode + 1:2d}"
            f" | Reward: {total_reward:.0f}"
        )

        if total_reward > best_reward:

            best_reward = total_reward

            best_frames = frames.copy()

    env.close()

    imageio.mimsave(
        "assets/gameplay.gif",
        best_frames,
        fps=20
    )

    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)
    print(f"Average Reward : {np.mean(rewards):.2f}")
    print(f"Best Reward    : {np.max(rewards):.0f}")
    print(f"Worst Reward   : {np.min(rewards):.0f}")
    print("=" * 70)
    print("Best Gameplay GIF Saved:")
    print("assets/gameplay.gif")


if __name__ == "__main__":
    play()