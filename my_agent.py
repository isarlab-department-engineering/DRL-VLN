import torch.multiprocessing as mp
import torch
import copy
import time
import numpy as np
from IMPALA_NAV.environment import Environment
import random
from torch.nn.utils.rnn import pad_sequence


def init_hidden():
	init_h = torch.nn.Parameter(torch.zeros(1, 1, 256).type(torch.FloatTensor), requires_grad=False)
	init_h_dir = torch.nn.Parameter(torch.zeros(1, 1, 256).type(torch.FloatTensor), requires_grad=False)
	return init_h, init_h_dir


class MyAgent(mp.Process):

	def __init__(self, gnet, idx, global_ep, wins, total_rewards, res_queue, queue, g_que, gamma, up_step, bs, n_actions, tokenizer, rep_freq):
		super(MyAgent, self).__init__()
		self.daemon = True
		self.idx = idx
		self.rep_freq = rep_freq
		self.global_ep, self.res_queue, self.queue, self.g_que, self.gamma, self.up_step, self.wins = global_ep, res_queue, queue, g_que, gamma, up_step, wins
		self.loss, self.vl, self.pl, self.cl, self.dl, self.ml, self.grad_norm = 0, 0, 0, 0, 0, 0, 0
		self.lnet = copy.deepcopy(gnet)
		self.rewards, self.rewards_neg, self.rewards_i, self.personal_reward = 0, 0, 0, 0
		self.bs = bs
		self.n_actions = n_actions
		self.tokenizer = tokenizer
		self.total_rewards = total_rewards
		self.lr = 0

	def step(self, reward, image, h, h_dir, instruction_embedding, att):
		with self.total_rewards.get_lock():
			self.total_rewards.value += reward
		with self.global_ep.get_lock():
			self.global_ep.value += 1
		action, h, h_dir, logits, _ = self.lnet.choose_action(image, h, h_dir, instruction_embedding, att)
		self.rewards += reward if reward > 0 else 0
		self.rewards_neg += reward if reward < 0 else 0
		self.personal_reward += reward
		return action, h, h_dir, logits

	def push_and_pull(self, bd, s_, bs, ba, br, h, bl, b_depth, b_instruction, instruction_, batt, att_, h_dir):
		seq_lengths = torch.LongTensor(list(map(len, b_instruction)))
		self.queue.put([torch.cat(bs), torch.tensor(ba), s_, bd, h, torch.tensor(br).unsqueeze(1), torch.stack(bl), torch.cat(b_depth), pad_sequence(b_instruction), seq_lengths, instruction_, torch.tensor(batt).unsqueeze(1), att_, h_dir])
		g_dict, self.loss, self.vl, self.pl, self.cl, self.dl, self.grad_norm, self.lr = self.g_que.get()
		self.lnet.load_state_dict(g_dict)

	def run(self):

		torch.manual_seed(self.idx)
		torch.cuda.manual_seed(self.idx)
		np.random.seed(self.idx)
		random.seed(self.idx)
		torch.backends.cudnn.deterministic = True

		env = Environment(9734 + self.idx, tokenizer=self.tokenizer, no_env=False, image=False, value=False, render=False, toy=False, neg_rew=True, att_loss=True)

		reward = 0
		sample_count = 0
		d = 0
		buffer_a, buffer_r, buffer_l, buffer_d, buffer_obs, buffer_i, buffer_h, buffer_h_dir, buffer_depth, buffer_instr, buffer_att = (), (), (), (), (), (), (), (), (), (), ()
		h, h_dir = init_hidden()
		h_ = copy.deepcopy(h)
		h_dir_ = copy.deepcopy(h_dir)
		n_step = 0
		obs, depth, instruction_, att = env.reset()
		instruction = instruction_
		instruction_embedding = self.lnet.get_embedding(instruction_, 1).detach()
		#extra_ch = (instruction_[-1] - 20).view(1, 1, 1, 1).repeat(1, 1, 84, 84).type(torch.FloatTensor)
		#obs = torch.cat((obs, extra_ch), dim=1)
		#obs = obs ** (instruction_[-1] - 18).type(torch.FloatTensor)
		#obs = obs if instruction_[att] == 2 else obs ** (instruction_[att] - 18).type(torch.FloatTensor)

		for p in self.lnet.parameters():
			p.requires_grad = False

		while self.global_ep.value < 1000000000:
			n_step += 1
			sample_count += 1

			action, h_, h_dir_, logits = self.step(reward, obs, h_, h_dir_, instruction_embedding, torch.ones((1)).type(torch.FloatTensor) if instruction_[att] == 2 else (instruction_[att] - 18).type(torch.FloatTensor))
			reward, obs_, depth_, att_ = env.env_step(action)
			#extra_ch = (instruction_[-1] - 20).view(1, 1, 1, 1).repeat(1, 1, 84, 84).type(torch.FloatTensor)
			#obs_ = torch.cat((obs_, extra_ch), dim=1)
			# obs_ = obs_ ** (instruction_[-1] - 18).type(torch.FloatTensor)
			#obs_ = obs_ if instruction_[att_] == 2 else obs_ ** (instruction_[att_] - 18).type(torch.FloatTensor)
			#if instruction[1] == 21:
			#	reward = 0.01 if action == 2 else -0.01
			#else:
			#	reward = 0.01 if action == 1 else -0.01

			if reward == 1 or reward < 0 or n_step % 900 == 0:
				obs_, depth_, instruction_, att_ = env.reset()
				#extra_ch = (instruction_[-1] - 20).view(1, 1, 1, 1).repeat(1, 1, 84, 84).type(torch.FloatTensor)
				#obs_ = torch.cat((obs_, extra_ch), dim=1)
				# obs_ = obs_ ** (instruction_[-1] - 18).type(torch.FloatTensor)
				#obs_ = obs_ if instruction_[att_] == 2 else obs_ ** (instruction_[att_] - 18).type(torch.FloatTensor)
				instruction_embedding = self.lnet.get_embedding(instruction_, 1).detach()
				reward = -1 if reward < 0 else reward
				if n_step % 900 == 0:
					d = True

			if len(buffer_obs) < 500:
				buffer_obs += (obs,)
				buffer_depth += (depth,)
				buffer_a += (action,)
				buffer_r += (reward,)
				buffer_instr += (instruction,)
				buffer_l += (logits,)
				buffer_d += (d,)
				buffer_h += (h_,)
				buffer_h_dir += (h_dir_,)
				buffer_att += (att,)
			else:
				buffer_obs = buffer_obs[1:] + (obs,)
				buffer_depth = buffer_depth[1:] + (depth,)
				buffer_instr = buffer_instr[1:] + (instruction,)
				buffer_a = buffer_a[1:] + (action,)
				buffer_r = buffer_r[1:] + (reward,)
				buffer_l = buffer_l[1:] + (logits,)
				buffer_d = buffer_d[1:] + (d,)
				buffer_h = buffer_h[1:] + (h_,)
				buffer_h_dir = buffer_h_dir[1:] + (h_dir_,)
				buffer_att = buffer_att[1:] + (att,)

			if sample_count == self.up_step or d and len(buffer_obs) >= self.up_step:
				for _ in range(self.rep_freq):
					if len(buffer_obs) == self.up_step:
						seq_lengths = torch.LongTensor(list(map(len, buffer_instr)))
						self.queue.put([torch.cat(buffer_obs), torch.tensor(buffer_a), obs_, buffer_d, buffer_h[-self.up_step], torch.tensor(buffer_r).unsqueeze(1), torch.stack(buffer_l), torch.cat(buffer_depth), pad_sequence(buffer_instr), seq_lengths, instruction_, torch.tensor(buffer_att).unsqueeze(1), att_, buffer_h_dir[-self.up_step]])
					else:
						replay_index = torch.randint(self.up_step + 1, len(buffer_obs), (1,))
						seq_lengths = torch.LongTensor(list(map(len, buffer_instr[-replay_index: -replay_index + self.up_step])))
						self.queue.put([torch.cat(buffer_obs[-replay_index: -replay_index + self.up_step]), torch.tensor(buffer_a[-replay_index: -replay_index + self.up_step]), buffer_obs[-replay_index + (self.up_step + 1)], buffer_d[-replay_index: -replay_index + self.up_step], buffer_h[-replay_index], torch.tensor(buffer_r[-replay_index: -replay_index + self.up_step]).unsqueeze(1), torch.stack(buffer_l[-replay_index: -replay_index + self.up_step]), torch.cat(buffer_depth[-replay_index: -replay_index + self.up_step]), pad_sequence(buffer_instr[-replay_index: -replay_index + self.up_step]), seq_lengths, buffer_instr[-replay_index + (self.up_step + 1)], torch.tensor(buffer_att[-replay_index: -replay_index + 100]).unsqueeze(1), buffer_att[-replay_index + (self.up_step + 1)], buffer_h_dir[-replay_index]])
				self.push_and_pull(buffer_d[-self.up_step:], obs_, buffer_obs[-self.up_step:], buffer_a[-self.up_step:], buffer_r[-self.up_step:], h, buffer_l[-self.up_step:], buffer_depth[-self.up_step:], buffer_instr[-self.up_step:], instruction_, buffer_att[-100:], att_, h_dir)
				sample_count = 0
				if d:
					print('Agent %i, step %i' % (self.idx, n_step))
					self.res_queue.put([self.rewards, self.global_ep.value, self.loss / self.bs, self.vl / self.bs, self.pl / self.bs, self.cl / (self.bs * self.n_actions * self.up_step), self.dl / self.bs, self.ml / self.bs, self.grad_norm, self.total_rewards.value, self.wins.value, self.lr, self.rewards_i, self.personal_reward, self.idx, self.rewards_neg])
					self.rewards, self.rewards_neg, self.rewards_i, self.personal_reward = 0, 0, 0, 0
					h, h_dir = init_hidden()
					d = 0
				h_ = copy.deepcopy(h)
				h_dir_ = copy.deepcopy(h_dir)
				instruction_embedding = self.lnet.get_embedding(instruction_, 1).detach()

			obs = obs_
			depth = depth_
			instruction = instruction_
			att = att_

		self.res_queue.put(None)
		self.queue.put(None)
		time.sleep(1)
		env.close_connection()
		print('Agent %i finished after %i steps.' % (self.idx, n_step))
