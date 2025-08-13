
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs("graphs", exist_ok=True)

EP_RE = re.compile(r"Episode\s+(\d+)\s*\|\s*Steps:\s*(\d+).*?Reward:\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE)
OPT_RE = re.compile(r"Option\s+(\d+):\s*avg len\s*=\s*([0-9]*\.?[0-9]+),\s*count\s*=\s*(\d+)", re.IGNORECASE)

def parse_log(log_path: str):
    """
    Parse a training log and return:
      - episodes: DataFrame with columns [episode, steps, reward, switches]
      - per_episode_option: dict[episode] -> dict[option] -> {'avg_len': float, 'count': int}
    """
    episodes = []
    per_episode_option = defaultdict(lambda: defaultdict(lambda: {'avg_len': 0.0, 'count': 0}))
    with open(log_path, 'r') as f:
        current_ep = None
        for line in f:
            ep_m = EP_RE.search(line)
            if ep_m:
                ep = int(ep_m.group(1))
                steps = int(ep_m.group(2))
                reward = float(ep_m.group(3))
                episodes.append({'episode': ep, 'steps': steps, 'reward': reward, 'switches': 0})
                current_ep = ep
                continue
            opt_m = OPT_RE.search(line)
            if opt_m and current_ep is not None:
                opt = int(opt_m.group(1))
                avg_len = float(opt_m.group(2))
                count = int(opt_m.group(3))
                per_episode_option[current_ep][opt]['avg_len'] = avg_len
                per_episode_option[current_ep][opt]['count'] = count
                # we'll fill 'switches' once we finish parsing
    # fill switches per episode
    ep_df = pd.DataFrame(episodes).sort_values('episode').reset_index(drop=True)
    for i, row in ep_df.iterrows():
        ep = int(row['episode'])
        switches = sum(per_episode_option[ep][o]['count'] for o in per_episode_option[ep])
        ep_df.at[i, 'switches'] = switches
    return ep_df, per_episode_option

def compute_option_stats(per_episode_option):
    """
    Compute global (weighted) option stats across the full run.
    Returns DataFrame with columns:
      option, total_terminations, total_time, weighted_avg_len, pct_time
    """
    totals = {}
    grand_time = 0.0
    for ep, od in per_episode_option.items():
        for o, vals in od.items():
            c = int(vals['count'])
            t = float(vals['avg_len']) * c
            if o not in totals:
                totals[o] = {'total_terminations': 0, 'total_time': 0.0}
            totals[o]['total_terminations'] += c
            totals[o]['total_time'] += t
            grand_time += t
    rows = []
    for o in sorted(totals):
        term = totals[o]['total_terminations']
        time = totals[o]['total_time']
        wavg = (time / term) if term > 0 else 0.0
        pct = (time / grand_time) * 100.0 if grand_time > 0 else 0.0
        rows.append({'option': o,
                     'total_terminations': term,
                     'total_time': time,
                     'weighted_avg_len': wavg,
                     'pct_time': pct})
    return pd.DataFrame(rows)

def plot_weighted_option_lengths(stats_df, outpath="graphs/tcoc_option_lengths_weighted.png"):
    plt.figure(figsize=(8,5))
    plt.bar(stats_df['option'], stats_df['weighted_avg_len'])
    plt.xlabel("Option Index")
    plt.ylabel("Average Option Duration (weighted)")
    plt.title("TC-OC: Average Duration of Options (weighted by terminations)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_option_time_share(stats_df, outpath="graphs/tcoc_option_time_share.png"):
    plt.figure(figsize=(8,5))
    plt.bar(stats_df['option'], stats_df['pct_time'])
    plt.xlabel("Option Index")
    plt.ylabel("Time Share (%)")
    plt.title("TC-OC: Percent of Time Spent in Each Option")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_switches(ep_df, outpath="graphs/tcoc_switches_per_episode.png", smooth_window=25):
    s = ep_df['switches'].rolling(smooth_window, min_periods=1).mean()
    plt.figure(figsize=(10,4))
    plt.plot(ep_df['episode'], s, label=f"Switches (smoothed, w={smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Number of Switches")
    plt.title("TC-OC: Option Switches per Episode")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_episode_length(ep_df, outpath="graphs/tcoc_episode_length.png", smooth_window=25):
    s = ep_df['steps'].rolling(smooth_window, min_periods=1).mean()
    plt.figure(figsize=(10,4))
    plt.plot(ep_df['episode'], s, label=f"Episode Length (smoothed, w={smooth_window})")
    plt.xlabel("Episode"); plt.ylabel("Steps")
    plt.title("TC-OC: Episode Length Over Time")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_success_rate(ep_df, outpath="graphs/tcoc_success_rate.png", window=50):
    # Define success as reward > 0
    success = (ep_df['reward'] > 0).astype(int)
    s = success.rolling(window, min_periods=1).mean()
    plt.figure(figsize=(10,4))
    plt.plot(ep_df['episode'], s, label=f"Success Rate (rolling mean, w={window})")
    plt.xlabel("Episode"); plt.ylabel("Success Rate")
    plt.title("TC-OC: Success Rate Over Time")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def run_all(log_path: str, save_prefix="graphs/tcoc"):
    ep_df, per_episode_option = parse_log(log_path)
    stats_df = compute_option_stats(per_episode_option)

    # Save tables
    ep_df.to_csv(f"{save_prefix}_episodes.csv", index=False)
    stats_df.to_csv(f"{save_prefix}_option_stats.csv", index=False)

    # Plots
    plot_weighted_option_lengths(stats_df, outpath=f"{save_prefix}_option_lengths_weighted.png")
    plot_option_time_share(stats_df, outpath=f"{save_prefix}_option_time_share.png")
    plot_switches(ep_df, outpath=f"{save_prefix}_switches_per_episode.png")
    plot_episode_length(ep_df, outpath=f"{save_prefix}_episode_length.png")
    plot_success_rate(ep_df, outpath=f"{save_prefix}_success_rate.png")

    return ep_df, stats_df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to training log file (e.g., logger.log)")
    ap.add_argument("--prefix", type=str, default="graphs/tcoc", help="Output prefix for files")
    args = ap.parse_args()
    run_all(args.log, save_prefix=args.prefix)
