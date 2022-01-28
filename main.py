import argparse
import pickle
from collections import deque
import copy
import datetime
import random
import os


import gym
import pybullet_envs
import json
from pathlib import Path
import numpy as np

from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction
from tqc.functions import eval_policy


EPISODE_LENGTH = 1000


class Logger:
    def __init__(self, experiment_folder, config):
        if os.path.exists(experiment_folder):
            raise Exception('Experiment folder exists, apparent seed conflict!')
        os.makedirs(experiment_folder)
        self.metrics_file = experiment_folder / "metrics.json"
        self.metrics_file.touch()
        with open(experiment_folder / 'config.json', 'w') as config_file:
            json.dump(config.__dict__, config_file)

        self._keep_n_episodes = 5
        self.exploration_episode_lengths = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_returns = deque(maxlen=self._keep_n_episodes)
        self.exploration_episode_number = 0

    def log(self, metrics):
        metrics['Exploration episodes number'] = self.exploration_episode_number
        for name, d in zip(['episode length', 'episode return'], [self.exploration_episode_lengths, self.exploration_episode_returns]):
            metrics[f'Exploration {name}, mean'] = np.mean(d)
            metrics[f'Exploration {name}, std'] = np.std(d)
        with open(self.metrics_file, 'a') as out_metrics:
            json.dump(metrics, out_metrics)
            out_metrics.write('\n')

    def update_evaluation_statistics(self, episode_length, episode_return):
        self.exploration_episode_number += 1
        self.exploration_episode_lengths.append(episode_length)
        self.exploration_episode_returns.append(episode_return)


def main(args, experiment_folder):
    # --- Init ---
    logger = Logger(experiment_folder, args)

    # remove TimeLimit
    env = gym.make(args.env).unwrapped
    eval_env = gym.make(args.env).unwrapped

    env = RescaleAction(env, -1., 1.)
    eval_env = RescaleAction(eval_env, -1., 1.)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = structures.ReplayBuffer(state_dim, action_dim, gamma=args.discount,
                                            n_episodes_to_store=args.Q_G_n_episodes, q_g_rollout_length=args.Q_G_rollout_length)
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim, args.n_quantiles, args.n_nets).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets

    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=top_quantiles_to_drop,
                      discount=args.discount,
                      tau=args.tau,
                      target_entropy=-np.prod(env.action_space.shape).item(),
                      sampling_scheme=args.sampling_scheme,
                      Q_G_eval_interval=args.Q_G_eval_interval,
                      Q_G_n_per_episode=args.Q_G_n_per_episode,
                      delta_gamma=args.delta_gamma,
                      d_update_interval=args.d_update_interval)

    evaluations = []
    state, done = env.reset(), False
    episode_return = 0
    episode_timesteps = 0
    episode_num = 0

    actor.train()
    full_checkpoints = [int(args.max_timesteps / 3), int(args.max_timesteps * 2 / 3), int(args.max_timesteps) - 1]
    if args.load_model is not None:
        start_iter = int(args.load_model.split('/')[-1].split('_')[1]) + 1
        trainer.load(args.load_model)
        replay_buffer = pickle.load(open(f'{args.load_model}_replay', 'rb'))
    else:
        start_iter = 0
    for t in range(start_iter, int(args.max_timesteps)):
        action = actor.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_timesteps += 1

        ep_end = done or episode_timesteps >= EPISODE_LENGTH
        replay_buffer.add(state, action, next_state, reward, done, ep_end)

        state = next_state
        episode_return += reward

        # Train agent after collecting sufficient data
        if t >= args.batch_size:
            step_metrics = trainer.train(replay_buffer, args.batch_size)
        else:
            step_metrics = dict()
        step_metrics['Timestamp'] = str(datetime.datetime.now())

        if done or episode_timesteps >= EPISODE_LENGTH:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            logger.update_evaluation_statistics(episode_timesteps, episode_return)
            # Reset environment
            state, done = env.reset(), False

            episode_return = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            threshold_metrics = trainer.eval_thresholds(replay_buffer, args.Q_G_n_per_episode)
            step_metrics['Total_timesteps'] = t + 1
            step_metrics['Evaluation_returns'] = eval_policy(actor, eval_env, EPISODE_LENGTH)
            step_metrics.update(threshold_metrics)
            logger.log(step_metrics)

        if t in full_checkpoints:
            trainer_save_name = f'{experiment_folder}/iter_{t}'
            trainer.save(trainer_save_name)
            with open(f'{experiment_folder}/iter_{t}_replay', 'wb') as outF:
                pickle.dump(replay_buffer, outF)
            # Remove previous checkpoint?
        elif (t + 1) % args.light_checkpoint_freq == 0:
            trainer_save_name = f'{experiment_folder}/iter_{t}'
            trainer.light_save(trainer_save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Humanoid-v3")          # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=1000, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=1, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--log_dir", default='data')
    parser.add_argument("--exp_name", default='experiment')
    parser.add_argument("--light_checkpoint_freq", type=int, default=100000)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--Q_G_eval_interval", type=int, default=50)
    parser.add_argument("--Q_G_n_episodes", type=int, default=50)
    parser.add_argument("--Q_G_n_per_episode", type=int, default=20)
    parser.add_argument("--sampling_scheme", type=str, choices=['uniform', 'episodes'], default='episodes')
    parser.add_argument("--Q_G_rollout_length", type=int, default=500)
    parser.add_argument("--delta_gamma", type=float, default=0.99)
    parser.add_argument("--d_update_interval", type=int, default=50000)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if args.seed is None:
        args.seed = random.randint(0, 1000000)

    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    exp_folder = log_dir / f'{args.exp_name}_{start_time}_{args.seed}'

    main(args, exp_folder)
