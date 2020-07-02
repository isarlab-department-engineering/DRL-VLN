import torch
import torch.nn as nn
import torch.nn.functional as F


CHANNELS = 3

D1 = 16
D2 = 32

DL = 256
DR = 256
DI = 256

NEW_SIZE = 81

MAX_LENGTH = 50


class Net(nn.Module):
	def __init__(self, a_dim, vocabulary_size, embedding_size, weights):
		super(Net, self).__init__()

		self.a_dim = a_dim
		self.attn_weights = None
		self.direct = None
		self.embedding_size = embedding_size

		self.conv1_dir = nn.Conv2d(in_channels=CHANNELS, out_channels=D1, kernel_size=8, stride=4, padding=0)
		self.drop1_dir = nn.Dropout2d()
		self.bnc1_dir = nn.GroupNorm(int(D1 / 2), D1)

		self.conv2_dir = nn.Conv2d(in_channels=D1, out_channels=D2, kernel_size=4, stride=2, padding=0)
		self.drop2_dir = nn.Dropout2d()
		self.bnc2_dir = nn.GroupNorm(int(D2 / 2), D2)

		self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size, padding_idx=0)
		self.embedding.weight = nn.Parameter(torch.FloatTensor(weights))
		self.embedding.weight.requires_grad = False

		self.gru_instr = nn.GRU(embedding_size, DI, bidirectional=True)
		self.drop3 = nn.Dropout()

		self.lin_dir = nn.Linear(NEW_SIZE * D2 + DI * 2, DL)

		self.gru_dir = nn.GRU(DL, DR)

		self.direction = nn.Linear(DR, 3)

		self.distribution = torch.distributions.Categorical

	def forward(self, x, h_dir, instruction, actor=False, seq_lengths=None):

		self.embedding.weight.requires_grad = False

		x1_dir = self.bnc1_dir(F.relu(self.drop1_dir(self.conv1_dir(x))))

		x2_dir = self.bnc2_dir(F.relu(self.drop2_dir(self.conv2_dir(x1_dir))))

		if actor:
			x_instr_1 = instruction
		else:
			x_instr_1 = self.get_embedding(instruction, x.shape[0], seq_lengths=seq_lengths)

		x3_dir = F.relu(self.lin_dir(torch.cat((x2_dir.view(-1, D2 * NEW_SIZE), x_instr_1), dim=1)))

		x4_dir, h_dir = self.gru_dir(x3_dir.view(-1, 1, DL), h_dir)

		x4_dir = F.relu(x4_dir.view(-1, DR))

		self.direct = F.softmax(self.direction(x4_dir), dim=-1)

		return h_dir, self.direct

	def get_embedding(self, instruction, bs, seq_lengths=None):
		instruction = instruction.view(bs, -1)
		x_instr = self.embedding(instruction).view(-1, bs, self.embedding_size)
		if seq_lengths is not None:
			x_instr = torch.nn.utils.rnn.pack_padded_sequence(x_instr, seq_lengths.cpu().numpy(), enforce_sorted=False)
			x_instr_1 = self.gru_instr(x_instr)[0]
			x_instr_1, _ = torch.nn.utils.rnn.pad_packed_sequence(x_instr_1)
			idx = (torch.LongTensor(seq_lengths) - 1).view(-1, 1).expand(len(seq_lengths), x_instr_1.size(2)).unsqueeze(0).cuda()
			x_instr_2 = x_instr_1.gather(0, torch.autograd.Variable(idx)).squeeze(0)
			x_instr_2 = F.relu(self.drop3(x_instr_2))
		else:
			x_instr_1 = self.gru_instr(x_instr)[0]
			x_instr_2 = F.relu(self.drop3(x_instr_1[-1]))
		return x_instr_2
