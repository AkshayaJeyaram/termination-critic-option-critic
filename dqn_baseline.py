import os
import numpy as np
import torch
import gym
from envs.four_rooms_env import GridFourRooms
from dqn_agent import DQNAgent

# Ensure log directory exists
os.makedirs("dqn_logs", exist_ok=True)

# Create environment
env = GridFourRooms()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(
    state_size=obs_dim,
    action_size=action_dim,
    hidden_size=128,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_size=10000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=10000,
    target_update=500
)

num_episodes = 2000
max_steps = 1000

# Logging file
reward_log_path = "dqn_logs/fourrooms_rewards.csv"
with open(reward_log_path, "w") as f:
    f.write("episode,reward\n")

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.store_transition((state, action, reward, next_state, float(done)))
        agent.train_step()
        agent.update_epsilon()

        state = next_state
        total_reward += reward

        if done:
            break

    # Print and log
    print(f"Episode {episode} | Reward: {total_reward:.2f}")
    with open(reward_log_path, "a") as f:
        f.write(f"{episode},{total_reward:.2f}\n")

env.close()
