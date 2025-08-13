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

