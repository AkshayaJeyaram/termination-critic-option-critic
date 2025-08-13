import random
import numpy as np
from collections import deque


class ExperienceBuffer:
    """
    A fixed-size buffer that stores environment transitions for experience replay.
    Each entry consists of (state, option, reward, next_state, done).
    """
    def __init__(self, max_size, seed=42):
        self.memory = deque(maxlen=max_size)
        self.rng = random.Random(seed)

    def store(self, state, option, reward, next_state, done):
        """Add a new transition to the buffer."""
        self.memory.append((state, option, reward, next_state, done))

    def sample_batch(self, batch_size):
        """Randomly sample a batch of transitions."""
        batch = self.rng.sample(self.memory, batch_size)
        states, options, rewards, next_states, dones = zip(*batch)
        return np.stack(states), options, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.memory)
