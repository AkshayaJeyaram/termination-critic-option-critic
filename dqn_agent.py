import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, gamma=0.99,
                 lr=1e-3, batch_size=64, buffer_size=10000, epsilon_start=1.0,
                 epsilon_end=0.1, epsilon_decay=10000, target_update=500):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = self._build_net(hidden_size).to(self.device)
        self.target_net = self._build_net(hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = deque(maxlen=buffer_size)

        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
        self.target_update = target_update

    def _build_net(self, hidden_size):
        return nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size)
        )

    def act(self, state):
        self.step_count += 1
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

    def update_epsilon(self):
        decay_rate = (self.epsilon - self.epsilon_final) / self.epsilon_decay
        self.epsilon = max(self.epsilon_final, self.epsilon - decay_rate)

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def sample_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states).to(self.device),
            torch.LongTensor(actions).unsqueeze(1).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        )

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            targets = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
