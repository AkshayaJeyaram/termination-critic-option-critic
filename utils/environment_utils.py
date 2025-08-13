import torch
import numpy as np
import gym
from envs.four_rooms_env import GridFourRooms


def make_env(env_name):
    """
    Creates and returns the environment.
    """
    is_atari = False
    if env_name.lower() == "fourrooms":
        env = GridFourRooms()
    else:
        env = gym.make(env_name)
        is_atari = True
    return env, is_atari


def to_tensor(ndarray, device='cpu', dtype=torch.float32):
    """
    Converts NumPy arrays (or lists) to torch tensors.
    """
    if isinstance(ndarray, list):
        ndarray = np.array(ndarray)
    return torch.tensor(ndarray, dtype=dtype).to(device)
