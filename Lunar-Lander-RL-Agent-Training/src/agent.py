import random

import torch
import torch.nn as nn
import torch.optim as optim

from src.network import QNetwork
from src.replay_buffer import ReplayBuffer


class DDQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = 0.99
        self.learning_rate = 1e-3
        self.batch_size = 128

        self.warmup_size = 5000

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.target_update_every = 1000
        self.train_step_count = 0

        self.memory = ReplayBuffer(100_000)

        self.policy_net = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_net = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()
    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state)

        return torch.argmax(q_values, dim=1).item()

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    def train(self):

        if len(self.memory) < max(self.batch_size, self.warmup_size):
            return

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(self.batch_size)

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        current_q = self.policy_net(states).gather(
            1,
            actions
        )

        with torch.no_grad():

            best_actions = self.policy_net(
                next_states
            ).argmax(
                dim=1,
                keepdim=True
            )

            next_q = self.target_net(
                next_states
            ).gather(
                1,
                best_actions
            )

            target_q = rewards + (
                self.gamma
                * next_q
                * (1 - dones)
            )

        loss = self.loss_fn(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()
        self.train_step_count += 1

        if self.train_step_count % self.target_update_every == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )
    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon * self.epsilon_decay,
            self.epsilon_min
        )