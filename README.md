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

## Plots and Diagnostics

### Auto-detect the latest -TC run:

```bash
PYTHONPATH=. python -m tcoc.visuals_tcoc
```

### Or point to a specific run directory:

```bash
PYTHONPATH=. python -m tcoc.visuals_tcoc --run-dir runs/OptionCriticMLP-fourrooms-default-TC
```

Outputs go to graphs/ (gitignored), e.g.:

- tcoc-<run>_reward_curve.png
  
- tcoc-<run>_loss_curves.png

- tcoc-<run>_termination_probs.png
  
- tcoc-<run>_beta_critic_loss.png
  
- tcoc-<run>_option_lengths_weighted.png
  
- tcoc-<run>_option_time_share.png
  
- tcoc-<run>_switches_per_episode.png
  
- tcoc-<run>_success_rate.png

CSVs: tcoc-<run>_episodes.csv, tcoc-<run>_option_stats.csv.

## Project Structure

```
termination_critic_OC_dissertation/
├── agents/
│   └── option_critic.py              # OC & TC-OC networks (MLP + CNN; β-adv head)
├── buffers/
│   └── experience_buffer.py          # Replay buffer
├── learners/
│   └── gradients.py                  # Actor, critic, and β-critic losses
├── utils/
│   ├── environment_utils.py          # Env factory (FourRooms/Atari), tensor helper
│   └── logging_utils.py              # Step CSV + episode text logs
├── envs/
│   └── four_rooms_env.py             # FourRooms gridworld
├── train/
│   └── train_option_critic.py        # Training entrypoint & CLI flags
├── tcoc/
│   ├── visuals_tcoc.py               # Plots for TC-OC (+ PPO/DQN comparisons)
│   └── option_diagnostics.py         # Parses episode logs → durations/usage
├── dqn_agent.py   dqn_baseline.py    # DQN baseline + logs (dqn_logs/)
├── ppo_baseline.py                   # PPO baseline + logs (ppo_logs/)
├── runs/                             # Per-run logs (CSV + text)  [gitignored]
├── graphs/                           # Generated plots            [gitignored]
├── models/                           # Checkpoints                [gitignored]
├── requirements.txt
└── README.md

```

## Training Details

Switch-goal behaviour:
When --switch-goal True on FourRooms, train/train_option_critic.py will:

~Episode 1000: save a checkpoint and call env.switch_goal().

~Episode 2000: save a final checkpoint and stop the run.

This yields ~2000 episodes total with a mid-run goal change.

## Logging

Per-step CSV: runs/<run>/training_metrics.csv
Columns include: steps, actor_loss, critic_loss, entropy, epsilon, beta_* (termination probs), beta_critic_loss.

Per-episode text: runs/<run>/session_log.log
Human-readable episode summaries and per-option stats, e.g.:

```bash
Episode k | Steps: ... | Reward: ... | ...
  - Option 0: avg len = 3.76, count = 25
  - Option 1: avg len = 5.10, count = 22
```

## Key Arguments

* --use-termination-critic (bool): enable TC-OC (β-advantage head + β-critic regression)

* --beta-critic-coefficient (float): weight for β-critic regression loss

* --num-options (int): number of options (default: 2)

* --min-option-duration (int): minimum intra-option duration before switching

* --entropy-reg, --entropy-decay: intra-option entropy annealing

* --beta-entropy-coeff (float): entropy regularization on β to avoid saturation

* --switch-goal (bool, FourRooms): mid-run goal switch & early stop

Standard OC knobs: --gamma, --learning-rate, --epsilon-*, --update-every, --target-update-freq, etc.

## Baselines and Comparisons

- DQN: PYTHONPATH=. python dqn_baseline.py → dqn_logs/fourrooms_rewards.csv
  
- PPO: PYTHONPATH=. python ppo_baseline.py → ppo_logs/fourrooms_rewards.csv
  
- tcoc/visuals_tcoc.py overlays TC-OC vs PPO vs DQN when those CSVs exist.

## References

- Pierre-Luc Bacon, Jean Harb, Doina Precup (2017). The Option-Critic Architecture.
- Follow-up work exploring learned termination/advantage signals for β (“termination critic”).
