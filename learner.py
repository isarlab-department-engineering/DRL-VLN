import torch
import torch.multiprocessing as mp
import math
import torch.nn.functional as F
import numpy as np
import random


class Learner(mp.Process):
	def __init__(self, g_n, que_i, que_o, n, global_ep, gamma, lr, up_step, length, bs, entropy_cost, baseline_cost, rep_freq):
		super(Learner, self).__init__()
		self.daemon = True
		self.rep_freq = rep_freq
		self.gnet = g_n
		self.queue_i = que_i
		self.queue_o = que_o
		self.n = n
		self.global_ep = global_ep
		self.gamma = gamma
		self.lr = lr
		self.up_step = up_step
		self.length = length
		self.bs = bs
		self.entropy_cost, self.baseline_cost = entropy_cost, baseline_cost

	def run(self):
		count = 0
		n = 0
		self.gnet.cuda()
		params = self.gnet.parameters()
		opt = torch.optim.RMSprop(params, lr=self.lr, eps=.1)

		torch.manual_seed(0)
		torch.cuda.manual_seed(0)
		np.random.seed(0)
		random.seed(0)
		torch.backends.cudnn.deterministic = True
		
		while True:
			n += self.bs
			opt.zero_grad()
			pl, vl, cl, dl, loss = torch.zeros((1)), torch.zeros((1)), torch.zeros((1)), torch.zeros((1)), torch.zeros((1))
			for i in range((self.rep_freq + 1) * self.bs):
				rg = self.queue_i.get()
				if rg is None:
					count += 1
					if count == self.n:
						break
				else:
					self.gnet.cuda()
					s = rg[0].unsqueeze(1).cuda()
					a = rg[1].unsqueeze(1).type(torch.IntTensor)
					s_ = rg[2].unsqueeze(0).cuda()
					d = torch.tensor(rg[3]).unsqueeze(1).type(torch.FloatTensor)
					h = rg[4].cuda()
					r = rg[5].type(torch.FloatTensor)
					r = Learner.clip_rewards(r)
					l = rg[6].unsqueeze(1)
					depth = rg[7].cuda()
					instruction = rg[8].cuda()
					seq_lengths = rg[9]
					instruction_ = rg[10].cuda()
					batt = rg[11].type(torch.FloatTensor)
					att_ = rg[12]
					h_dir = rg[13].cuda()

					directions = torch.stack([instruction[v, i] - 2 if instruction[v, i] == 2 else instruction[v, i] - 19 for i, v in enumerate(batt.squeeze().type(torch.IntTensor))], dim=0).unsqueeze(1)

					self.gnet.train()
					logits, values, h, h_dir, values_curiosity, d_pred, dir_preds = self.gnet(s, h, h_dir, instruction, directions.type(torch.FloatTensor).cuda().squeeze() + 1, seq_lengths=seq_lengths)
					logits = logits.view(-1, h.shape[1], logits.shape[-1])

					self.gnet.eval()
					_, bootstrap_value, _, _, _, _, _ = self.gnet(s_, h, h_dir, instruction_, torch.ones((1)).cuda() if instruction_[att_] == 2 else (instruction_[att_] - 18).type(torch.FloatTensor).cuda(), dir_preds[-1].unsqueeze(0))
					bootstrap_value = bootstrap_value.squeeze().cpu() * (1 - d[-1])

					probs = torch.clamp(F.softmax(logits, dim=-1), 0.000001, 0.999999)
					m = torch.distributions.Categorical(probs)

					discounts = (1 - d) * self.gamma

					vs, pg_advantages = Learner.v_trace(probs.cpu(), l, a, bootstrap_value, values.cpu(), r, discounts)

					p_, v_, c_, l_ = self.get_loss(a, pg_advantages, m, vs, values, probs)

					pl += p_
					vl += v_
					cl += c_
					loss += l_

					lr = self.lr
					l_.backward()
					l_.detach_()

					torch.nn.utils.clip_grad_norm_(self.gnet.parameters(), 400)
					grad_norm = 0
					for gp in self.gnet.parameters():
						if gp.grad is not None:
							grad_norm += gp.grad.pow(2).sum()
					grad_norm = math.sqrt(grad_norm)

					if grad_norm != grad_norm:
						opt.zero_grad()
						print('grad_norm nan')

			opt.step()

			loss = loss.cpu() / (self.rep_freq + 1)
			vl, pl, cl, dl = vl.cpu() / (self.rep_freq + 1), pl.cpu() / (self.rep_freq + 1), cl.cpu() / (self.rep_freq + 1), dl.cpu() / (self.rep_freq + 1)
			loss.detach_(), vl.detach_(), pl.detach_(), cl.detach_(), dl.detach_()

			g = self.gnet.cpu()
			while not self.queue_o.empty():
				try:
					self.queue_o.get(timeout=0.01)
				except:
					pass
			for b in range(self.bs):
				self.queue_o.put([g.state_dict(), loss, vl, pl, cl, dl, grad_norm, lr])

			if n % (10000 * self.bs / self.up_step) == 0:
				torch.save(self.gnet.state_dict(), '/path_to_model/model.pt')

	@staticmethod
	def v_trace(probs, bl, ba, bootstrap_value, values, br, discounts):

		m = torch.distributions.Categorical(probs)

		clip_rho_threshold = 1
		clip_pg_rho_threshold = 1

		b_probs = torch.clamp(F.softmax(bl, dim=-1), 0.000001, 0.999999)
		b_m = torch.distributions.Categorical(b_probs)

		target_action_log_probs = m.log_prob(ba)
		behaviour_action_log_probs = b_m.log_prob(ba)

		log_rhos = target_action_log_probs - behaviour_action_log_probs
		rhos = torch.exp(log_rhos)
		clipped_rhos = torch.clamp(rhos, 0, clip_rho_threshold)
		clipped_pg_rhos = torch.clamp(rhos, 0, clip_pg_rho_threshold)

		values_t_plus_1 = torch.cat((values[1:], bootstrap_value.unsqueeze(0)))

		deltas = clipped_rhos * (br + discounts * values_t_plus_1 - values)

		acc = 0
		dt = []
		for i in reversed(range(len(deltas))):
			acc = deltas[i] + discounts[i]*clipped_rhos[i]*acc
			dt.append(acc)

		vs_minus_v_xs = torch.stack(dt).flip(0)
		vs = (vs_minus_v_xs + values)

		vs_t_plus_1 = torch.cat((vs[1:], bootstrap_value.unsqueeze(0)))
		pg_advantages = clipped_pg_rhos * (br + discounts * vs_t_plus_1 - values)

		return vs.detach(), pg_advantages.detach()

	def get_loss(self, ba, pg_advantages, m, vs, values, probs):
		pl = (-m.log_prob(ba.cuda()) * pg_advantages.cuda()).sum()
		vl = 0.5 * (vs.cuda() - values).pow(2).sum()
		cl = (probs * - torch.log(probs)).sum()
		return pl, vl, cl, pl + self.baseline_cost * vl - self.entropy_cost * cl

	@staticmethod
	def clip_rewards(br):
		squeezed = torch.tanh(br / 5.0)
		squeezed = torch.where(br < 0, .3 * squeezed, squeezed) * 5.
		return squeezed
