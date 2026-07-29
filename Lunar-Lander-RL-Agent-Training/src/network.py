import torch.nn as nn
class QNetwork(nn.Module):
    """
    Simple fully-connected Q-network.

    Input  : state (8 continuous values for LunarLander)
    Output : Q-value for each of the 4 discrete actions
    """

    def __init__(self, state_size, action_size):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, action_size)
        )

    def forward(self, x):
        return self.net(x)