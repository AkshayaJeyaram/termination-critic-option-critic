# Termination-Critic Option-Critic (TC-OC)

A clean PyTorch implementation of **Option-Critic** augmented with a **Termination Critic** (β-advantage head + β-critic regression), plus vanilla OC, DQN, and PPO baselines, reproducible logs, and one-shot plotting/diagnostics.

---

## Quickstart

```bash
# 0) (recommended) create a clean env
pyenv install 3.10.13
pyenv virtualenv 3.10.13 oc-env
pyenv activate oc-env
pip install -r requirements.txt

# 1) Train TC-OC on FourRooms with a mid-run goal switch (~2000 episodes total)
PYTHONPATH=. python train/train_option_critic.py \
  --use-termination-critic True \
  --beta-critic-coefficient 1.0 \
  --switch-goal True \
  --env fourrooms --num-options 2 --seed 0
```

## Baselines

```bash
# DQN
PYTHONPATH=. python dqn_baseline.py

# PPO
PYTHONPATH=. python ppo_baseline.py
```

##Plots and Diagnostics

###Auto-detect the latest -TC run:

```bash
PYTHONPATH=. python -m tcoc.visuals_tcoc
```

###Or point to a specific run directory:

```bash
PYTHONPATH=. python -m tcoc.visuals_tcoc --run-dir runs/OptionCriticMLP-fourrooms-default-TC
```

Outputs go to graphs/ (gitignored), e.g.:

  tcoc-<run>_reward_curve.png
  
  tcoc-<run>_loss_curves.png
  
  tcoc-<run>_termination_probs.png
  
  tcoc-<run>_beta_critic_loss.png
  
  tcoc-<run>_option_lengths_weighted.png
  
  tcoc-<run>_option_time_share.png
  
  tcoc-<run>_switches_per_episode.png
  
  tcoc-<run>_success_rate.png

CSVs: tcoc-<run>_episodes.csv, tcoc-<run>_option_stats.csv.



