import torch
import torch.nn as nn
import torch.nn.functional as F
from IMPALA_INSTR_ATT.model_instr_dir import Net as NetDir

CHANNELS = 3

D1 = 16
D2 = 32

DL = 256
DR = 256
DI = 256

#NEW_SIZE = 196
NEW_SIZE = 81
#NEW_SIZE = 70

MAX_LENGTH = 50


class Net(nn.Module):
	def __init__(self, a_dim, vocabulary_size, embedding_size, weights):
		super(Net, self).__init__()

		self.a_dim = a_dim
		self.direct = None

		self.net_dir = NetDir(a_dim, vocabulary_size, embedding_size, weights)

		self.conv1 = nn.Conv2d(in_channels=CHANNELS, out_channels=D1, kernel_size=8, stride=4, padding=0)
		self.drop1 = nn.Dropout2d()
		self.bnc1 = nn.GroupNorm(int(D1 / 2), D1)

		self.conv2 = nn.Conv2d(in_channels=D1, out_channels=D2, kernel_size=4, stride=2, padding=0)
		self.drop2 = nn.Dropout2d()
		self.bnc2 = nn.GroupNorm(int(D2 / 2), D2)

		self.lin = nn.Linear(NEW_SIZE * D2, DL)

		self.gru = nn.GRU(DL, DR)

		self.p = nn.Linear(DR, a_dim + 1)

		self.distribution = torch.distributions.Categorical

	def forward(self, x, h, h_dir, instruction, att, depth=None, actor=False, seq_lengths=None):

		x_84 = F.adaptive_avg_pool2d(x.view(-1, CHANNELS, x.shape[-2], x.shape[-1]), 84)

		h_dir, self.direct = self.net_dir(x_84, h_dir, instruction, actor, seq_lengths)

		#####################################################################################################

		#x_84 = x_84 ** att.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		att = (torch.argmax(self.direct, -1) + 1).type(torch.FloatTensor).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		x_84 = x_84 ** att if actor and False else x_84 ** att.cuda()

		x1 = self.bnc1(F.relu(self.drop1(self.conv1(x_84))))

		x2 = self.bnc2(F.relu(self.drop2(self.conv2(x1))))

		x3 = F.relu(self.lin(x2.view(-1, D2 * NEW_SIZE)))

		x4, h = self.gru(x3.view(-1, 1, DL), h)

		s0 = x4.shape[0]
		s1 = x4.shape[1]

		x4 = F.relu(x4.view(-1, DR))

		x5 = self.p(x4).view(s0, s1, self.a_dim + 1)
		logits = x5[:, :, :self.a_dim]
		values = x5[:, :, -1].view(s0, s1)

		if depth is not None:
			return logits.squeeze(), values, h, h_dir, None, None, self.direct

		return logits.squeeze(), values, h, h_dir, None, x2, x3

	def get_embedding(self, instruction, bs, seq_lengths=None):
		return self.net_dir.get_embedding(instruction, bs, seq_lengths)

	def choose_action(self, s, h, h_dir, instruction, att, train=False):
		if not train:
			self.eval()
		logits, values, h, h_dir, _, x2, x3 = self.forward(s, h, h_dir, instruction, att, actor=True)
		#logits, values, h, _ = self.forward(s, h, vis_match, get_conv_out=train)
		probs = torch.clamp(F.softmax(logits, dim=-1), 0.00001, 0.99999).data
		m = self.distribution(probs)
		action = m.sample().type(torch.IntTensor)
		if train:
			return action, h.data, h_dir.data, logits, values, x2, x3
		return action, h.data, h_dir.data, logits, values

	def choose_action1(self, s, h, instruction):
		self.eval()
		logits, values, h, _ = self.forward(s, h, instruction)
		probs = torch.clamp(F.softmax(logits, dim=-1), 0.00001, 0.99999).data
		return torch.argmax(probs, -1), h.data, logits, values

	def get_weights(self):
		layers = [self.conv1, self.bnc1, self.conv2, self.bnc2, self.deconv1, self.debnc1, self.deconv2, self.lin_match, self.match_softmax, self.lstm_instr, self.lin, self.lstm, self.p, self.v]
		weigths = []
		for layer in layers:
			tot = 0
			for p in layer.parameters():
				tot += p.sum()
			weigths.append(tot.item())
		return weigths