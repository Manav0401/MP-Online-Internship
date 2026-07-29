import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        # Shared Feature Extractor
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )

    def forward(self, x):

        features = self.feature(x)

        value = self.value_stream(features)

        advantage = self.advantage_stream(features)

        q_values = value + (
            advantage - advantage.mean(dim=1, keepdim=True)
        )

        return q_values