# utils/logging_utils.py
import os
import csv
import time
from collections import deque
from statistics import mean
from colorama import Fore, init

init(autoreset=True)

class Logger:
    def __init__(self, logdir="runs", run_name=None):
        self.run_name = run_name or f"session_{int(time.time())}"
        self.logdir = os.path.join(logdir, self.run_name)
        os.makedirs(self.logdir, exist_ok=True)

        self.episode_count = 0
        # Base fields; more columns can be added dynamically via **extra
        self.metric_fields = ["steps", "actor_loss", "critic_loss", "entropy", "epsilon"]
        self.metrics_file = os.path.join(self.logdir, "training_metrics.csv")
        self.log_file = os.path.join(self.logdir, "session_log.log")
        self.recent_rewards = deque(maxlen=100)

        # initialize empty file with header
        with open(self.metrics_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.metric_fields)
            writer.writeheader()

        with open(self.log_file, "w") as logf:
            logf.write(f"=== Logging initiated at {time.ctime()} ===\n")

    def _rewrite_with_new_fields(self, new_fields):
        """If new columns appear, rewrite CSV header once to include them."""
        # read all existing rows
        with open(self.metrics_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # extend fields
        for k in new_fields:
            if k not in self.metric_fields:
                self.metric_fields.append(k)

        # rewrite file with new header and backfilled rows
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.metric_fields)
            writer.writeheader()
            for r in rows:
                # ensure all keys exist
                for k in self.metric_fields:
                    r.setdefault(k, None)
                writer.writerow(r)

    def log_data(self, steps, actor_loss, critic_loss, entropy, epsilon, **extra):
        """
        Log per-step metrics. Pass additional columns via **extra
        (e.g., beta_0=..., beta_1=..., beta_critic_loss=...).
        """
        # if new keys appear, upgrade the CSV header first
        new_cols = [k for k in extra.keys() if k not in self.metric_fields]
        if new_cols:
            self._rewrite_with_new_fields(new_cols)

        def as_num(x):
            try:
                return float(x.item())
            except Exception:
                try:
                    return float(x)
                except Exception:
                    return x

        entry = {
            "steps": steps,
            "actor_loss": as_num(actor_loss) if actor_loss is not None else None,
            "critic_loss": as_num(critic_loss) if critic_loss is not None else None,
            "entropy": as_num(entropy) if entropy is not None else None,
            "epsilon": as_num(epsilon) if epsilon is not None else None,
        }
        # coerce any extras into plain numbers where possible
        for k, v in extra.items():
            entry[k] = as_num(v) if v is not None else None

        with open(self.metrics_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.metric_fields)
            writer.writerow(entry)

    def log_episode(self, steps, episode_reward, option_lengths, episode_steps, epsilon, current_option=None):
        self.episode_count += 1
        self.recent_rewards.append(episode_reward)
        avg_reward = mean(self.recent_rewards)

        summary = (f"{Fore.GREEN}Episode {self.episode_count} | Steps: {steps:,} | "
                   f"Reward: {episode_reward:.2f} | Avg Reward (100): {avg_reward:.2f} | "
                   f"Ep Steps: {episode_steps} | Epsilon: {epsilon:.3f}")

        if current_option is not None:
            summary += f" | Active Option: {current_option}"

        print(summary)
        with open(self.log_file, "a") as logf:
            logf.write(summary + "\n")

        for option_id, durations in option_lengths.items():
            if durations:
                average_duration = sum(durations) / len(durations)
                detail = f"  - Option {option_id}: avg len = {average_duration:.2f}, count = {len(durations)}"
                print(detail)
                with open(self.log_file, "a") as logf:
                    logf.write(detail + "\n")

        if self.episode_count % 50 == 0:
            print()
