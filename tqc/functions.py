import numpy as np

import torch

from tqc import DEVICE


def eval_policy(policy, eval_env, max_episode_steps, eval_episodes=10):
    policy.eval()
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        t = 0
        while not done and t < max_episode_steps:
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            t += 1
    avg_reward /= eval_episodes
    policy.train()
    return avg_reward

def eval_bias(policy, critic, eval_env, gamma, max_episode_steps, extra_steps, eval_episodes, target_entropy, alpha):
    policy.train()
    all_values = []
    all_returns = []
    for _ in range(eval_episodes):
        ep_rewards = []
        ep_states = []
        ep_actions = []
        state, done = eval_env.reset(), False
        t = 0
        while not done and t < max_episode_steps + extra_steps:
            action = policy.select_action(state)
            ep_states.append(state)
            ep_actions.append(action)
            state, reward, done, _ = eval_env.step(action)
            ep_rewards.append(reward)
            t += 1

        cur_return = 0
        returns = []
        for r in ep_rewards[::-1]:
            cur_return = r + alpha * target_entropy + cur_return * gamma
            returns.append(cur_return)
        returns = returns[::-1]

        n_to_use = min(max_episode_steps, t)
        returns_to_use = np.array(returns[:n_to_use])
        all_returns.append(returns_to_use)

        states_to_use = np.stack(ep_states[:n_to_use])
        states_to_use_torch = torch.FloatTensor(states_to_use).to(DEVICE)
        actions_to_use = np.stack(ep_actions[:n_to_use])
        actions_to_use_torch = torch.FloatTensor(actions_to_use).to(DEVICE)
        with torch.no_grad():
            values = critic(states_to_use_torch, actions_to_use_torch).mean(axis=2).mean(axis=1)
            values = values.cpu().numpy()
        all_values.append(values)
    return all_values, all_returns


def quantile_huber_loss_f(quantiles, samples, samples_mask=None):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss)
    if samples_mask is not None:
        samples_mask = samples_mask.unsqueeze(0).unsqueeze(0)
        loss = (loss * samples_mask).sum(-1) / samples_mask.sum()
    loss = loss.mean()
    return loss
