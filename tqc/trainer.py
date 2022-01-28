import numpy as np
import torch
from math import ceil, floor

from tqc.functions import quantile_huber_loss_f
from tqc import DEVICE


class Trainer(object):
	def __init__(
		self,
		*,
		actor,
		critic,
		critic_target,
		discount,
		tau,
		top_quantiles_to_drop,
		target_entropy,
		sampling_scheme,
		Q_G_eval_interval,
		Q_G_n_per_episode,
		delta_gamma,
		d_update_interval,
	):
		self.actor = actor
		self.critic = critic
		self.critic_target = critic_target
		self.log_alpha = torch.zeros((1,), requires_grad=True, device=DEVICE)

		# TODO: check hyperparams
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

		self.discount = discount
		self.tau = tau
		self.top_quantiles_to_drop = top_quantiles_to_drop
		self.target_entropy = target_entropy

		self.quantiles_total = critic.n_quantiles * critic.n_nets
		self.samples_mask = torch.zeros(1, self.quantiles_total, dtype=torch.float32, device=DEVICE)
		self.calculate_quantile_mask_2()

		self.total_it = 0
		self.Q_G_eval_interval = Q_G_eval_interval
		self.Q_G_n_per_episode = Q_G_n_per_episode
		self.Q_G_delta = 0
		self.sampling_scheme = sampling_scheme
		self.delta_gamma = delta_gamma
		self.d_update_interval = d_update_interval

	def calculate_quantile_mask(self):
		top_quantiles_to_drop = self.log_eta.sigmoid() * self.quantiles_total
		self.samples_mask = torch.zeros(1, self.quantiles_total, dtype=torch.float64, device=DEVICE)
		for i in range(self.quantiles_total):
			mask = self.quantiles_total - top_quantiles_to_drop - i
			self.samples_mask[0, i] = max(min(mask, 1), 0)

	def calculate_quantile_mask_2(self):
		top_quantiles_to_drop = self.top_quantiles_to_drop
		top = ceil(top_quantiles_to_drop)
		bot = floor(top_quantiles_to_drop)
		self.samples_mask[0, 0:self.quantiles_total - top] = 1
		self.samples_mask[0, self.quantiles_total - bot:] = 0
		if top != bot:
			self.samples_mask[0, self.quantiles_total - top] = top - top_quantiles_to_drop

	def add_next_z_metrics(self, metrics, next_z):
		for t in range(1, self.critic.n_quantiles + 1):
			total_quantiles_to_keep = t * self.critic.n_nets
			metrics[f'Target_Q/Q_value_t={t}'] = next_z[:, :total_quantiles_to_keep].mean().__float__()

	def train(self, replay_buffer, batch_size=256):
		metrics = dict()
		state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
		alpha = torch.exp(self.log_alpha)
		metrics['alpha'] = alpha.item()
		metrics['top_quantiles_to_drop'] = self.top_quantiles_to_drop

		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(next_state)

			# compute and cut quantiles at the next state
			next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
			sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
			self.add_next_z_metrics(metrics, sorted_z)

			# compute target
			target = reward + not_done * self.discount * (sorted_z - alpha * next_log_pi)

		cur_z = self.critic(state, action)
		critic_loss = quantile_huber_loss_f(cur_z, target, self.samples_mask)
		metrics['critic_loss'] = critic_loss.item()

		# --- Policy and alpha loss ---
		new_action, log_pi = self.actor(state)
		metrics['actor_entropy'] = - log_pi.mean().item()
		alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
		actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()
		metrics['actor_loss'] = actor_loss.item()

		# --- Update ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		if self.total_it > 10000 and (self.total_it + 1) % self.Q_G_eval_interval == 0:
			self.eval_thresholds(replay_buffer, self.Q_G_n_per_episode)

		if self.total_it > 10000 and (self.total_it + 1) % self.d_update_interval == 0:
			self.update_d()

		self.total_it += 1
		return metrics

	def eval_thresholds_by_type(self, replay_buffer, n_per_episode, sampling_scheme):
		res = dict()
		alpha = torch.exp(self.log_alpha)
		if sampling_scheme == 'uniform':
			states, actions, returns, bs_states, bs_multiplier = replay_buffer.gather_returns_uniform(self.discount, float(alpha), n_per_episode)
		elif sampling_scheme == 'episodes':
			states, actions, returns, bs_states, bs_multiplier = replay_buffer.gather_returns(self.discount, float(alpha), n_per_episode)
		else:
			raise Exception("No such sampling scheme")
		tail_actions = self.actor(bs_states)[0]
		tail_z = self.critic_target(bs_states, tail_actions)
		tail_z = tail_z.reshape(tail_z.shape[0], -1)
		tail_z = tail_z.mean(1, keepdim=True) * bs_multiplier * np.power(replay_buffer.gamma, replay_buffer.q_g_rollout_length)
		res[f'LastReplay_{sampling_scheme}/Returns'] = (returns + tail_z).mean().__float__()

		cur_z = self.critic(states, actions)
		cur_z = cur_z.reshape(cur_z.shape[0], -1)
		cur_z = cur_z.sort(dim=1)[0]
		for t in range(1, self.critic.n_quantiles + 1):
			total_quantiles_to_keep = t * self.critic.n_nets
			res[f'LastReplay_{sampling_scheme}/Q_value_t={t}'] = cur_z[:, :total_quantiles_to_keep].mean().__float__()
		return res

	def eval_thresholds(self, replay_buffer, n_per_episode):
		res_uniform = self.eval_thresholds_by_type(replay_buffer, n_per_episode, 'uniform')
		res_episodes = self.eval_thresholds_by_type(replay_buffer, n_per_episode, 'episodes')
		res = dict()
		res.update(res_uniform)
		res.update(res_episodes)
		last_Q_G_delta = res[f'LastReplay_{self.sampling_scheme}/Q_value_t={self.critic.n_quantiles}'] - \
						 res[f'LastReplay_{self.sampling_scheme}/Returns']
		self.Q_G_delta = self.Q_G_delta * self.delta_gamma + last_Q_G_delta * (1 - self.delta_gamma)
		return res

	def update_d(self):
		if self.Q_G_delta < 0 and self.top_quantiles_to_drop > 0:
			self.top_quantiles_to_drop -= 1
		elif self.Q_G_delta > 0 and self.top_quantiles_to_drop < self.quantiles_total:
			self.top_quantiles_to_drop += 1
		self.calculate_quantile_mask_2()

	def save(self, filename):
		filename = str(filename)
		self.light_save(filename)
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

	def light_save(self, filename):
		filename = str(filename)
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_target.state_dict(), filename + "_critic_target")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.log_alpha, filename + "_log_alpha")

	def load(self, filename):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.log_alpha = torch.load(filename + '_log_alpha')
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
		self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))
