from collections import deque, namedtuple
import random
import numpy as np


# Experience tuple
Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    Stores past experiences and samples random mini-batches.
    """

    def __init__(self, capacity: int):

        self.memory = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        transition = Transition(
            state,
            action,
            reward,
            next_state,
            done
        )

        self.memory.append(transition)

    def sample(self, batch_size: int):

        batch = random.sample(self.memory, batch_size)

        states = np.asarray(
            [t.state for t in batch],
            dtype=np.float32
        )

        actions = np.asarray(
            [t.action for t in batch],
            dtype=np.int64
        )

        rewards = np.asarray(
            [t.reward for t in batch],
            dtype=np.float32
        )

        next_states = np.asarray(
            [t.next_state for t in batch],
            dtype=np.float32
        )

        dones = np.asarray(
            [t.done for t in batch],
            dtype=np.float32
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):

        return len(self.memory)