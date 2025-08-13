import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import os
from envs.four_rooms_env import GridFourRooms

# Ensure PPO log directory exists
os.makedirs("ppo_logs", exist_ok=True)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x)

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

def ppo_update(policy, optimizer, states, actions, log_probs_old, returns, advantages, clip_epsilon=0.2):
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    returns = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(advantages)
    log_probs_old = torch.FloatTensor(log_probs_old)

    for _ in range(4):  # 4 epochs
        logits, values = policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values.squeeze(), returns)

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = GridFourRooms()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = PolicyNetwork(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=2.5e-4)

    max_episodes = 2000
    gamma = 0.99
    lam = 0.95

    for episode in range(max_episodes):
        obs = env.reset()
        done = False

        states, actions, rewards, values, log_probs = [], [], [], [], []
        episode_reward = 0

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            logits, value = policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()

            next_obs, reward, done, _ = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob)

            obs = next_obs
            episode_reward += reward

        advantages = compute_advantages(rewards, values, gamma, lam)
        returns = [adv + val for adv, val in zip(advantages, values)]
        ppo_update(policy, optimizer, states, actions, log_probs, returns, advantages)

        # Log reward to console
        print(f"Episode {episode} | Reward: {episode_reward:.2f}")

        # Save reward to CSV
        with open("ppo_logs/fourrooms_rewards.csv", "a") as f:
            f.write(f"{episode},{episode_reward:.2f}\n")

if __name__ == "__main__":
    main()
