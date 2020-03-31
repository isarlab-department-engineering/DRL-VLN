import torch.backends.cudnn as cudnn
import os

cudnn.benchmark = True

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
	# from IMPALA_INSTR_ATT.model_instr import Net		# CAT
	from IMPALA_INSTR_ATT.model_instr_GAU import Net		# GAU
	# from IMPALA_INSTR_ATT.model_instr_GAU_2 import Net		# OURS
	from tensorboardX import SummaryWriter
	from IMPALA_INSTR_ATT.learner import Learner
	from IMPALA_INSTR_ATT.my_agent import MyAgent
	from IMPALA_INSTR_ATT.Tokenizer import Tokenizer
	from IMPALA_INSTR_ATT.GloVe import GloVe
	import torch.multiprocessing as mp
	import torch
	import numpy as np
	import random

	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	np.random.seed(0)
	random.seed(0)
	torch.backends.cudnn.deterministic = True

	N = 12
	GAMMA = 0.99
	UP_STEP = 100
	BS = 4
	LR = 0.0002		#0.0005
	ENTROPY_COST = 0.001		#0.00025
	BASELINE_COST = 0.5
	REP_FREQ = 2
	N_ACTIONS = 3

	tokenizer = Tokenizer()
	instruction_vocabulary = ['go', 'straight', 'at', 'the', 'next', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'intersections', 'intersection', 'turn', 'right', 'left']
	tokenizer.fit_on_texts(instruction_vocabulary)

	weights = GloVe(tokenizer).get_weights()
	vocabulary_size = weights.shape[0]
	embedding_size = weights.shape[1]

	mp.set_start_method('spawn')

	writer = SummaryWriter()

	gnet = Net(N_ACTIONS, vocabulary_size, embedding_size, weights)
	gnet.load_state_dict(torch.load('/home/andromeda/PycharmProjects/CustomUnreal/My_Agent/models/model_instruction_GAU_FINAL_3.pt'))

	global_ep, wins, tot_rewards = mp.Value('i', 0), mp.Value('i', 0), mp.Value('d', 0.)
	res_queue, queue, g_que = mp.Queue(), mp.Queue(), mp.Queue()

	learner = Learner(gnet, queue, g_que, N, global_ep, GAMMA, LR, UP_STEP, 1000000000, BS, ENTROPY_COST, BASELINE_COST, REP_FREQ)

	agents = [MyAgent(gnet, i, global_ep, wins, tot_rewards, res_queue, queue, g_que, GAMMA, UP_STEP, BS, N_ACTIONS, tokenizer, REP_FREQ) for i in range(N)]

	learner.start()

	[agent.start() for agent in agents]

	while 1:
		r = res_queue.get()
		if r is not None:
			writer.add_scalar('global_ep_r', r[0], r[1])
			writer.add_scalar('loss', r[2], r[1])
			writer.add_scalar('val_loss', r[3], r[1])
			writer.add_scalar('pol_loss', r[4], r[1])
			writer.add_scalar('H_loss', r[5], r[1])
			writer.add_scalar('depth_loss', r[6], r[1])
			writer.add_scalar('match_loss', r[7], r[1])
			writer.add_scalar('grad_norm', r[8], r[1])
			writer.add_scalar('total_reward', r[9], r[1])
			writer.add_scalar('wins', r[10], r[1])
			writer.add_scalar('lr', r[11], r[1])
			writer.add_scalar('intrinsic_rewards', r[12], r[1])
			writer.add_scalar('personal_reward_%i' % r[14], r[13], r[1])
			writer.add_scalar('global_ep_r_neg', r[15], r[1])
		else:
			break

	[agent.join() for agent in agents]
	learner.join()
