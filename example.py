import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from IMPALA_INSTR_ATT.model_instr_GAU_2 import Net
from IMPALA_NAV.environment import Environment
from IMPALA_INSTR_ATT.Tokenizer import Tokenizer
from IMPALA_INSTR_ATT.GloVe import GloVe

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

X = 224
Y = 224


def init_hidden():
	init_h = torch.nn.Parameter(torch.zeros(1, 1, 256).type(torch.FloatTensor), requires_grad=False).cuda()
	init_h_dir = torch.nn.Parameter(torch.zeros(1, 1, 256).type(torch.FloatTensor), requires_grad=False).cuda()
	return init_h, init_h_dir


def rormalize_image(obs):
	obs = NORMALIZE(obs).view(1, 3, X, Y)
	return obs


def load_networks(nav_path):
	tokenizer = Tokenizer()
	instruction_vocabulary = ['go', 'straight', 'at', 'the', 'next', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'intersections', 'intersection', 'turn', 'right', 'left']
	tokenizer.fit_on_texts(instruction_vocabulary)

	weights = GloVe(tokenizer).get_weights()
	vocabulary_size = weights.shape[0]
	embedding_size = weights.shape[1]

	instr_net = Net(3, vocabulary_size, embedding_size, weights).cuda()
	instr_net.load_state_dict(torch.load(nav_path))

	instr_net.eval()

	return instr_net


TIME = 3000
MAX_STEP = 100000
MAX_EP = 100

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

instr_net = load_networks(nav_path='/path_to_model/model.pt')

tokenizer = Tokenizer()
instruction_vocabulary = ['go', 'straight', 'at', 'the', 'next', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'intersections', 'intersection', 'turn', 'right', 'left']
tokenizer.fit_on_texts(instruction_vocabulary)

env = Environment(9733, tokenizer=tokenizer, no_env=False, test=True, image=False, render=True, toy=False, neg_rew=True, att_loss=True)

ep_reward = 0
reward = 0
h, h_dir = init_hidden()
obs, _, instruction, _ = env.reset()
X = obs.shape[2]
Y = obs.shape[1]

instruction_embedding = instr_net.get_embedding(instruction.cuda(), 1).detach()

obs = update_image(obs).cuda()

step = 0
ep = 0
while (reward != 1) and step < MAX_STEP and ep < MAX_EP:
	step += 1

	action, h, h_dir, logits, v = instr_net.choose_action(obs, h, h_dir, instruction_embedding, None)

	reward, obs_, _, _ = env.env_step(action)
	ep_reward += reward

	if reward == 1 or step % 900 == 0 or reward < 0:
		ep += 1
		obs_, _, instruction, _ = env.reset()
		instruction_embedding = instr_net.get_embedding(instruction.cuda(), 1).detach()
		h, h_dir = init_hidden()
		step = 0

	reward = 0

	obs = obs_.cuda()

	obs = rormalize_image(obs)

env.close_connection()
