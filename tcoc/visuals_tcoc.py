import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# -------------------------
# Run directory discovery
# -------------------------
def discover_latest_tcoc_run(runs_root="runs"):
    """
    Find the most recently modified run directory that looks like a TC-OC run.
    Preference order:
      1) any dir containing '-TC' in its name
      2) otherwise, any dir that contains a training_metrics.csv
    """
    if not os.path.isdir(runs_root):
        return None

    tc_dirs = []
    other_dirs = []
    for d in os.listdir(runs_root):
        full = os.path.join(runs_root, d)
        if not os.path.isdir(full):
            continue
        csv_path = os.path.join(full, "training_metrics.csv")
        mtime = os.path.getmtime(full)
        if "-TC" in d:
            tc_dirs.append((mtime, full))
        elif os.path.exists(csv_path):
            other_dirs.append((mtime, full))

    if tc_dirs:
        return sorted(tc_dirs, key=lambda x: x[0])[-1][1]
    if other_dirs:
        return sorted(other_dirs, key=lambda x: x[0])[-1][1]
    return None

def pick_log_path(run_dir):
    """Prefer session_log.log (your TC-OC code) but fall back to logger.log."""
    for candidate in ("session_log.log", "logger.log"):
        p = os.path.join(run_dir, candidate)
        if os.path.exists(p):
            return p
    # fallback path (likely missing but keeps downstream code simple)
    return os.path.join(run_dir, "session_log.log")

# -------------------------
# Helpers
# -------------------------
def smooth(data, window=100):
    return pd.Series(data).rolling(window, min_periods=1).mean()

def extract_rewards(log_path):
    """
    Parse 'Episode ... | Reward: X' lines from the log.
    Works with the format produced by utils.logging_utils.Logger.
    """
    rewards = []
    if not os.path.exists(log_path):
        return rewards
    with open(log_path, 'r') as f:
        for line in f:
            if "Episode" in line and "Reward" in line:
                try:
                    parts = line.split("|")
                    for p in parts:
                        if "Reward" in p:
                            # find first number (handles +/-, ints/floats)
                            m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", p)
                            if m:
                                rewards.append(float(m.group(1)))
                            break
                except Exception:
                    continue
    return rewards

def extract_option_stats(log_path):
    """
    Aggregate option usage from per-episode summaries like:
        "  - Option 0: avg len = 3.76, count = 25"
    Returns:
      options (sorted list),
      weighted_len (list),
      counts (list),
      time_used (list: avg_len * count per option)
    """
    opt_time = defaultdict(float)
    opt_terms = Counter()
    if not os.path.exists(log_path):
        return [], [], [], []
    with open(log_path, 'r') as f:
        for line in f:
            if "Option" in line and "avg len" in line and "count" in line:
                try:
                    opt = int(re.findall(r"Option\s+(\d+)", line)[0])
                    avg_len = float(re.findall(r"avg len\s*=\s*([0-9]*\.?[0-9]+)", line)[0])
                    count = int(re.findall(r"count\s*=\s*(\d+)", line)[0])
                    opt_time[opt] += avg_len * count
                    opt_terms[opt] += count
                except Exception:
                    continue
    options = sorted(opt_time.keys())
    weighted_len = [(opt_time[o] / opt_terms[o]) if opt_terms[o] > 0 else 0.0 for o in options]
    counts = [opt_terms[o] for o in options]
    time_used = [opt_time[o] for o in options]
    return options, weighted_len, counts, time_used

# -------------------------
# Plotters (guard legends)
# -------------------------
def plot_reward_curve(LOG_PATH, GRAPH_DIR, PREFIX):
    rewards = extract_rewards(LOG_PATH)
    if not rewards:
        print("No rewards parsed from log; skipping reward_curve.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(rewards, window=100), label="Episode Reward (smoothed)")
    plt.xlabel("Episodes"); plt.ylabel("Reward")
    plt.title("TC-OC: Smoothed Reward Curve")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{PREFIX}_reward_curve.png")); plt.close()

def plot_entropy(CSV_PATH, GRAPH_DIR, PREFIX):
    if not os.path.exists(CSV_PATH):
        print("training_metrics.csv not found; skipping entropy.")
        return
    df = pd.read_csv(CSV_PATH)
    if "entropy" not in df.columns:
        print("Column 'entropy' not found; skipping entropy.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(df["entropy"], window=500), label="Entropy (smoothed)")
    plt.xlabel("Steps"); plt.ylabel("Entropy")
    plt.title("TC-OC: Intra-Option Policy Entropy")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{PREFIX}_entropy_curve.png")); plt.close()

def plot_loss_curves(CSV_PATH, GRAPH_DIR, PREFIX):
    if not os.path.exists(CSV_PATH):
        print("training_metrics.csv not found; skipping loss curves.")
        return
    df = pd.read_csv(CSV_PATH)
    if not {"actor_loss", "critic_loss", "steps"}.issubset(df.columns):
        print("Loss columns not found; skipping loss curves.")
        return
    df = df.dropna(subset=["actor_loss", "critic_loss"])
    if df.empty:
        print("No non-NaN loss rows; skipping loss curves.")
        return
    window = 100
    df["actor_loss_smooth"] = df["actor_loss"].rolling(window, min_periods=1).mean()
    df["critic_loss_smooth"] = df["critic_loss"].rolling(window, min_periods=1).mean()
    plt.figure(figsize=(12, 5))
    plt.plot(df["steps"], df["actor_loss_smooth"], label="Actor Loss (smoothed)")
    plt.plot(df["steps"], df["critic_loss_smooth"], label="Critic Loss (smoothed)")
    plt.xlabel("Steps"); plt.ylabel("Loss")
    plt.title("TC-OC: Loss Curves")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{PREFIX}_loss_curves.png")); plt.close()

def plot_termination_probs(CSV_PATH, GRAPH_DIR, PREFIX):
    if not os.path.exists(CSV_PATH):
        print("training_metrics.csv not found; skipping termination probs.")
        return
    df = pd.read_csv(CSV_PATH)
    term_cols = [c for c in df.columns if c.startswith("beta_")]
    if not term_cols:
        print("No beta_* columns; skipping termination probs.")
        return
    plt.figure(figsize=(10, 5))
    plotted = False
    for col in term_cols:
        if df[col].notna().any():
            plt.plot(df["steps"], smooth(df[col], 200), label=col)
            plotted = True
    plt.xlabel("Steps"); plt.ylabel("β (termination probability)")
    plt.title("TC-OC: Termination Probability Trends")
    plt.grid(True)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{PREFIX}_termination_probs.png")); plt.close()

def plot_beta_critic_loss(CSV_PATH, GRAPH_DIR, PREFIX):
    if not os.path.exists(CSV_PATH):
        print("training_metrics.csv not found; skipping beta_critic_loss.")
        return
    df = pd.read_csv(CSV_PATH)
    if "beta_critic_loss" not in df.columns:
        print("beta_critic_loss not in CSV; skipping.")
        return
    s = df["beta_critic_loss"].rolling(200, min_periods=1).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(df["steps"], s, label="β-critic loss (smoothed)")
    plt.xlabel("Steps"); plt.ylabel("Loss"); plt.title("TC-OC: Termination-Critic Loss")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{PREFIX}_beta_critic_loss.png")); plt.close()

def plot_option_lengths_weighted(LOG_PATH, GRAPH_DIR, PREFIX):
    options, weighted_len, _, _ = extract_option_stats(LOG_PATH)
    if not options:
        print("No option stats found in log; skipping option_lengths_weighted.")
        return
    plt.figure(figsize=(8, 5))
    plt.bar(options, weighted_len)
    plt.xlabel("Option Index"); plt.ylabel("Average Option Duration (weighted)")
    plt.title("TC-OC: Average Duration of Options (weighted by terminations)")
    plt.grid(True, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{PREFIX}_option_lengths_weighted.png")); plt.close()

def plot_option_frequency_and_time_share(LOG_PATH, GRAPH_DIR, PREFIX):
    options, _, counts, time_used = extract_option_stats(LOG_PATH)
    if not options:
        print("No option stats found in log; skipping frequency/time-share.")
        return
    # Frequency
    plt.figure(figsize=(8, 5))
    plt.bar(options, counts)
    plt.xlabel("Option Index"); plt.ylabel("Times Selected (terminations)")
    plt.title("TC-OC: Option Selection Frequency")
    plt.grid(True, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{PREFIX}_option_frequency.png")); plt.close()

    # Time share
    total_time = sum(time_used) if time_used else 0.0
    pct = [(t / total_time) * 100.0 if total_time > 0 else 0.0 for t in time_used]
    plt.figure(figsize=(8, 5))
    plt.bar(options, pct)
    plt.xlabel("Option Index"); plt.ylabel("Time Share (%)")
    plt.title("TC-OC: Percent of Time Spent in Each Option")
    plt.grid(True, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{PREFIX}_option_time_share.png")); plt.close()

def extract_csv_rewards(csv_path, skip_header=True):
    rewards = []
    if not os.path.exists(csv_path):
        return rewards
    with open(csv_path, 'r') as f:
        if skip_header:
            next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    rewards.append(float(parts[1]))
                except ValueError:
                    continue
    return rewards

def plot_comparison_curve(LOG_PATH, GRAPH_DIR, PREFIX):
    oc_rewards = extract_rewards(LOG_PATH)
    ppo_rewards = extract_csv_rewards("ppo_logs/fourrooms_rewards.csv")
    dqn_rewards = extract_csv_rewards("dqn_logs/fourrooms_rewards.csv")
    if not (oc_rewards or ppo_rewards or dqn_rewards):
        print("No comparison curves available; skipping.")
        return
    plt.figure(figsize=(10, 5))
    if oc_rewards: plt.plot(smooth(oc_rewards, 100), label="TC-OC")
    if ppo_rewards: plt.plot(smooth(ppo_rewards, 100), label="PPO")
    if dqn_rewards: plt.plot(smooth(dqn_rewards, 100), label="DQN")
    plt.title("Reward Comparison: TC-OC vs PPO vs DQN")
    plt.xlabel("Episodes"); plt.ylabel("Smoothed Reward")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{PREFIX}_comparison_reward_curve_all.png")); plt.close()

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None,
                    help="Path to a specific run directory under 'runs/'. If omitted, auto-discovers the latest TC-OC run.")
    args = ap.parse_args()

    run_dir = args.run_dir or discover_latest_tcoc_run("runs")
    if run_dir is None:
        print("Could not find any run directory under 'runs/'. Exiting.")
        return

    csv_path = os.path.join(run_dir, "training_metrics.csv")
    log_path = pick_log_path(run_dir)
    graph_dir = "graphs"
    os.makedirs(graph_dir, exist_ok=True)
    prefix = f"tcoc-{os.path.basename(run_dir)}"

    print(f"[visuals_tcoc] Using run dir: {run_dir}")
    print(f"[visuals_tcoc] Using log file: {log_path}")
    print(f"[visuals_tcoc] Using metrics:  {csv_path}")

    # Plots
    plot_reward_curve(log_path, graph_dir, prefix)
    plot_entropy(csv_path, graph_dir, prefix)
    plot_loss_curves(csv_path, graph_dir, prefix)
    plot_termination_probs(csv_path, graph_dir, prefix)
    plot_beta_critic_loss(csv_path, graph_dir, prefix)
    plot_option_lengths_weighted(log_path, graph_dir, prefix)
    plot_option_frequency_and_time_share(log_path, graph_dir, prefix)
    plot_comparison_curve(log_path, graph_dir, prefix)

    print(f"   Plots saved under: {graph_dir} with prefix '{prefix}'")
    print(f"   Used run dir: {run_dir}")

if __name__ == "__main__":
    main()
