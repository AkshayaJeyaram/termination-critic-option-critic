import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
from math import exp

from utils.environment_utils import to_tensor


class OptionCriticMLP(nn.Module):
    """
    Multi-Layer Perceptron version of the Option-Critic agent.
    Handles Q-value estimation, intra-option policies, and termination predictions.
    """
    def __init__(self, input_dim, num_actions, num_options,
                 temperature=1.0, eps_start=1.0, eps_min=0.1, eps_decay=int(1e6),
                 eps_test=0.05, device='cpu', eval_mode=False):
        super().__init__()

        self.device = device
        self.eval_mode = eval_mode
        self.num_actions = num_actions
        self.num_options = num_options
        self.temperature = temperature

        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.step_count = 0

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU()
        )

        self.q_head = nn.Linear(64, num_options)
        self.termination_head = nn.Linear(64, num_options)
        # NEW: β-advantage head (for Termination-Critic)
        self.beta_adv_head = nn.Linear(64, num_options)

        self.intra_option_weights = nn.Parameter(torch.zeros(num_options, 64, num_actions))
        self.biases = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not eval_mode)

    def extract_features(self, obs):
        # obs: torch.Tensor expected
        obs = obs.to(self.device)
        if obs.dim() == 1:
            # single (D,) -> (1, D)
            obs = obs.unsqueeze(0)
        elif obs.dim() > 2:
            # e.g., accidentally passed (1, B, D) -> squeeze leading singleton
            obs = obs.squeeze(0)
        return self.feature_extractor(obs)

    def select_option_epsilon_greedy(self, features):
        if torch.rand(()) < self.epsilon:
            return int(torch.randint(0, self.num_options, ()).item())
        return int(self.compute_Q_values(features).argmax(dim=-1).item())

    def current_epsilon(self):
        # same formula without incrementing step_count
        if self.eval_mode:
            return self.eps_test
        return self.eps_min + (self.eps_start - self.eps_min) * exp(-self.step_count / self.eps_decay)

    def compute_Q_values(self, features):
        return self.q_head(features)

    def predict_termination_probs(self, features):
        return self.termination_head(features).sigmoid()

    # NEW: predict β-advantage A_beta(s,o) (no activation)
    def predict_beta_advantage(self, features):
        return self.beta_adv_head(features)

    def should_terminate(self, features, current_option):
        beta = self.termination_head(features)[:, current_option].sigmoid()
        # Clamp only for sampling stability; do NOT clamp in loss computations
        beta_clamped = beta.clamp(1e-4, 1 - 1e-4)
        terminate = Bernoulli(beta_clamped).sample()
        next_option = self.compute_Q_values(features).argmax(dim=-1)
        return bool(terminate.item()), next_option.item()

    def sample_action(self, features, option_idx):
        logits = features @ self.intra_option_weights[option_idx] + self.biases[option_idx]
        dist = Categorical(logits.div(self.temperature).softmax(dim=-1))
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def select_greedy_option(self, features):
        return self.compute_Q_values(features).argmax(dim=-1).item()

    @property
    def epsilon(self):
        if self.eval_mode:
            return self.eps_test
        eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.step_count / self.eps_decay)
        self.step_count += 1
        return eps


class OptionCriticCNN(nn.Module):
    """
    Convolutional Neural Network version of the Option-Critic agent.
    Suitable for image-based environments (e.g., Atari).
    """
    def __init__(self, input_channels, num_actions, num_options,
                 temperature=1.0, eps_start=1.0, eps_min=0.1, eps_decay=int(1e6),
                 eps_test=0.05, device='cpu', eval_mode=False):
        super().__init__()

        self.device = device
        self.eval_mode = eval_mode
        self.num_actions = num_actions
        self.num_options = num_options
        self.temperature = temperature

        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.step_count = 0

        conv_out_dim = 7 * 7 * 64

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out_dim, 512), nn.ReLU()
        )

        self.q_head = nn.Linear(512, num_options)
        self.termination_head = nn.Linear(512, num_options)
        # NEW: β-advantage head (for Termination-Critic)
        self.beta_adv_head = nn.Linear(512, num_options)

        self.intra_option_weights = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.biases = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not eval_mode)

    def extract_features(self, obs):
        obs = obs.to(self.device)
        if obs.dim() == 3:
            # single (C, H, W) -> (1, C, H, W)
            obs = obs.unsqueeze(0)
        elif obs.dim() == 5 and obs.size(0) == 1:
            # e.g., (1, B, C, H, W) -> (B, C, H, W)
            obs = obs.squeeze(0)
        # if obs.dim() == 4: already (B, C, H, W) -> fine
        return self.feature_extractor(obs)

    def select_option_epsilon_greedy(self, features):
        if torch.rand(()) < self.epsilon:
            return int(torch.randint(0, self.num_options, ()).item())
        return int(self.compute_Q_values(features).argmax(dim=-1).item())

    def current_epsilon(self):
        if self.eval_mode:
            return self.eps_test
        return self.eps_min + (self.eps_start - self.eps_min) * exp(-self.step_count / self.eps_decay)

    def compute_Q_values(self, features):
        return self.q_head(features)

    def predict_termination_probs(self, features):
        return self.termination_head(features).sigmoid()

    # NEW
    def predict_beta_advantage(self, features):
        return self.beta_adv_head(features)

    def should_terminate(self, features, current_option):
        beta = self.termination_head(features)[:, current_option].sigmoid()
        beta_clamped = beta.clamp(1e-4, 1 - 1e-4)
        terminate = Bernoulli(beta_clamped).sample()
        next_option = self.compute_Q_values(features).argmax(dim=-1)
        return bool(terminate.item()), next_option.item()

    def sample_action(self, features, option_idx):
        logits = features @ self.intra_option_weights[option_idx] + self.biases[option_idx]
        dist = Categorical(logits.div(self.temperature).softmax(dim=-1))
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def select_greedy_option(self, features):
        return self.compute_Q_values(features).argmax(dim=-1).item()

    @property
    def epsilon(self):
        if self.eval_mode:
            return self.eps_test
        eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.step_count / self.eps_decay)
        self.step_count += 1
        return eps
