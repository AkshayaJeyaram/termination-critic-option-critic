# train/train_option_critic.py
import argparse
import torch
import numpy as np
from copy import deepcopy
import os

from agents.option_critic import OptionCriticMLP, OptionCriticCNN
from buffers.experience_buffer import ExperienceBuffer
from learners.gradients import compute_actor_gradient, compute_critic_gradient
from utils.environment_utils import make_env, to_tensor
from utils.logging_utils import Logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).strip().lower()
    if v in ("yes", "true", "t", "1", "y", "on"):
        return True
    if v in ("no", "false", "f", "0", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


def train(args):
    env, is_atari = make_env(args.env)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    AgentClass = OptionCriticCNN if is_atari else OptionCriticMLP

    # Build agent (MLP for FourRooms; CNN if you point at an image env)
    if is_atari:
        agent = AgentClass(
            input_channels=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            num_options=args.num_options,
            temperature=args.temp,
            eps_start=args.epsilon_initial,
            eps_min=args.epsilon_final,
            eps_decay=args.epsilon_decay_steps,
            eps_test=args.optimal_eps,
            device=device
        )
    else:
        agent = AgentClass(
            input_dim=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            num_options=args.num_options,
            temperature=args.temp,
            eps_start=args.epsilon_initial,
            eps_min=args.epsilon_final,
            eps_decay=args.epsilon_decay_steps,
            eps_test=args.optimal_eps,
            device=device
        )

    target_agent = deepcopy(agent)
    optimizer = torch.optim.RMSprop(agent.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        env.seed(args.seed)
    except Exception:
        pass

    replay_buffer = ExperienceBuffer(max_size=args.max_history, seed=args.seed)

    run_tag = f"{AgentClass.__name__}-{args.env}-{args.exp or 'default'}"
    if args.use_termination_critic:
        run_tag += "-TC"
    logger = Logger(logdir=args.logdir, run_name=run_tag)

    total_steps = 0
    min_option_duration = args.min_option_duration

    if args.switch_goal and hasattr(env, "goal_index"):
        print(f"Current goal index {env.goal_index}")

    while total_steps < args.max_steps_total:
        obs = env.reset()
        state = agent.extract_features(to_tensor(obs))
        greedy_option = agent.select_greedy_option(state)
        current_option = agent.select_option_epsilon_greedy(state)

        option_lengths = {opt: [] for opt in range(args.num_options)}

        # Optional goal switch checkpointing (FourRooms)
        if args.switch_goal and logger.episode_count == 1000:
            ckpt_path = f"models/vanilla_oc_{args.env}_seed{args.seed}_ep2000.pth"
            payload = {'model_state_dict': agent.state_dict()}
            if hasattr(env, "goal_index"):
                payload['goal_state'] = env.goal_index
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(payload, ckpt_path)
            if hasattr(env, "switch_goal"):
                env.switch_goal()
                if hasattr(env, "goal_index"):
                    print(f"New goal {env.goal_index}")

        if args.switch_goal and logger.episode_count > 2000:
            ckpt_path = f"models/vanilla_oc_{args.env}_seed{args.seed}_ep4000.pth"
            payload = {'model_state_dict': agent.state_dict()}
            if hasattr(env, "goal_index"):
                payload['goal_state'] = env.goal_index
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(payload, ckpt_path)
            break

        done = False
        ep_steps = 0
        rewards = 0.0
        curr_op_len = 0
        option_terminated = True

        actor_loss = None
        critic_loss = None
        beta_critic_loss_val = None
        entropy = torch.tensor(0.0)

        while not done and ep_steps < args.max_steps_ep:
            # cache epsilon ONCE per step
            epsilon = agent.epsilon

            # switch option only when previous option terminated and min duration satisfied
            if option_terminated and curr_op_len >= min_option_duration:
                option_lengths[current_option].append(curr_op_len)
                current_option = agent.select_option_epsilon_greedy(state)
                curr_op_len = 0

            action, logp, entropy = agent.sample_action(state, current_option)
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.store(obs, current_option, reward, next_obs, done)
            rewards += reward

            if len(replay_buffer) > args.batch_size:
                # === Actor (policy + termination (+ optional β-critic reg)) ===
                actor_loss, beta_closs = compute_actor_gradient(
                    obs, current_option, logp, entropy, reward, done, next_obs,
                    agent, target_agent, args, total_steps
                )
                beta_critic_loss_val = beta_closs

                loss = actor_loss
                if beta_closs is not None:
                    # IMPORTANT: actually optimize against β-critic regression
                    loss = loss + args.beta_critic_coeff * beta_closs

                # === Critic (Q) ===
                if total_steps % args.update_every == 0:
                    batch = replay_buffer.sample_batch(args.batch_size)
                    critic_loss = compute_critic_gradient(agent, target_agent, batch, args)
                    loss = loss + critic_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()

                if total_steps % args.target_update_freq == 0:
                    target_agent.load_state_dict(agent.state_dict())

            state = agent.extract_features(to_tensor(next_obs))
            option_terminated, greedy_option = agent.should_terminate(state, current_option)

            total_steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            # β columns for CSV
            with torch.no_grad():
                term_probs_now = agent.predict_termination_probs(state)  # (1, num_options)
            beta_log = {f"beta_{i}": float(term_probs_now[0, i].item()) for i in range(args.num_options)}

            extras = {}
            if beta_critic_loss_val is not None:
                extras["beta_critic_loss"] = float(beta_critic_loss_val.item())

            ent_val = float(entropy.item()) if hasattr(entropy, "item") else float(entropy)
            logger.log_data(
                total_steps,
                actor_loss,
                critic_loss,
                ent_val,
                epsilon,
                **beta_log,
                **extras
            )

        logger.log_episode(total_steps, rewards, option_lengths, ep_steps, epsilon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration for training an Option-Critic agent")

    # Environment & logging
    parser.add_argument("--env", type=str, default="fourrooms", help="Target environment name")
    parser.add_argument("--exp", type=str, default=None, help="Optional experiment label")
    parser.add_argument("--logdir", type=str, default="runs", help="Directory for logging training runs")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generation")
    parser.add_argument("--cuda", type=str2bool, default=True, help="Enable CUDA if available")

    # Agent & learning
    parser.add_argument("--num-options", type=int, default=2, help="Number of options")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--learning-rate", type=float, default=0.0007, help="Optimizer learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--max-history", type=int, default=20000, help="Replay buffer size")

    # Exploration
    parser.add_argument("--epsilon-initial", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.1)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000)

    # Termination & entropy
    parser.add_argument("--temp", type=float, default=1.5, help="Softmax temperature")
    parser.add_argument("--termination-reg", type=float, default=0.012, help="Margin in termination objective")
    parser.add_argument("--entropy-reg", type=float, default=0.015, help="Intra-option entropy weight")
    parser.add_argument("--entropy-decay", type=float, default=8e4, help="Entropy weight decay steps")
    parser.add_argument("--min-option-duration", type=int, default=3, help="Min steps before switching option")
    parser.add_argument("--beta-entropy-coeff", type=float, default=0.0, help="Entropy reg on β")

    # Termination-Critic switches
    parser.add_argument("--use-termination-critic", type=str2bool, default=False,
                        help="Enable Termination-Critic (learn A_beta)")
    # Support both spellings; use same dest
    parser.add_argument("--beta-critic-coefficient", dest="beta_critic_coeff", type=float, default=1.0,
                        help="Weight for β-advantage regression loss")
    parser.add_argument("--beta-critic-coef", dest="beta_critic_coeff", type=float, default=None,
                        help="Alias; overrides coefficient if provided")

    # Training control
    parser.add_argument("--max-steps-ep", type=int, default=16000, help="Max steps per episode")
    parser.add_argument("--max-steps-total", type=int, default=4_000_000, help="Total training steps")
    parser.add_argument("--update-every", type=int, default=6, help="Update interval (steps)")
    parser.add_argument("--target-update-freq", type=int, default=400, help="Target sync interval")
    parser.add_argument("--switch-goal", type=str2bool, default=False, help="FourRooms: switch goal mid-run")
    parser.add_argument("--optimal-eps", type=float, default=0.05, help="Epsilon in eval mode")

    args = parser.parse_args()

    # If user passed the alias, use it
    if args.beta_critic_coeff is None:
        args.beta_critic_coeff = 1.0

    train(args)
