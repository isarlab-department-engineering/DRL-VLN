import os
import cv2
import csv
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torch import autograd
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


def update_val(val, v):
	val[:-1] = val[1:]
	val[-1] = v
	return val


def update_image(obs, cam):

	frame = np.moveaxis(obs.numpy(), 0, -1).astype(np.float64)

	obs = NORMALIZE(obs).view(1, 3, X, Y)

	frame_norm = np.moveaxis(obs.view(3, X, Y).numpy(), 0, -1).astype(np.float64)

	if torch.max(cam) != 0:
		cam = transforms.ToPILImage()(cam / torch.max(cam))
		cam = cam.resize((X, Y))
	else:
		cam = transforms.ToPILImage()(cam)
		cam = cam.resize((X, Y))

	alpha = np.zeros((X, Y, 3))
	alpha[:, :, 0] = cam
	alpha[:, :, 1] = cam
	alpha[:, :, 2] = cam

	frame_cam = (cv2.multiply(alpha / 255, frame)[..., ::-1] * 255).astype(np.uint8)
	frame = (frame * 255).astype(np.uint8)[..., ::-1]
	frame = cv2.resize(frame, None, fx=3, fy=3)
	frame_norm = (frame_norm * 255).astype(np.uint8)[..., ::-1]

	return obs, frame, frame_cam, frame_norm


def load_networks(nav_path, train):
	tokenizer = Tokenizer()
	instruction_vocabulary = ['go', 'straight', 'at', 'the', 'next', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'intersections', 'intersection', 'turn', 'right', 'left']
	tokenizer.fit_on_texts(instruction_vocabulary)

	weights = GloVe(tokenizer).get_weights()
	vocabulary_size = weights.shape[0]
	embedding_size = weights.shape[1]

	instr_net = Net(3, vocabulary_size, embedding_size, weights).cuda()
	instr_net.load_state_dict(torch.load(nav_path))

	if train:
		instr_net.train()
	else:
		instr_net.eval()

	return instr_net


def get_gradients(nav_net, logit, out):
	nav_net.zero_grad()
	gradients = autograd.grad(outputs=logit, inputs=out, grad_outputs=torch.ones_like(logit), only_inputs=True)[0]
	w_k = torch.mean(torch.mean(out * F.relu(gradients), -1), -1).unsqueeze(-1).unsqueeze(-1)
	cam = F.relu(torch.mean((w_k*out), 1))
	return cam.detach().cpu()


def get_att_frame(nav_net, instr):
	weights = nav_net.attn_weights[:, :len(instr)]
	weights = weights.detach().cpu().numpy()
	#print(instruction_vocabulary[instruction[np.argmax(weights[0])] - 1])
	weights = np.repeat(weights, len(instr), axis=0)
	weights = (weights * 255).astype(np.uint8)
	weights = cv2.resize(weights, (400, 30))

	weights_pad = nav_net.attn_weights[:, len(instr):]
	weights_pad = weights_pad.detach().cpu().numpy()
	weights_pad = np.repeat(weights_pad, len(instr), axis=0)
	weights_pad = (weights_pad * 255).astype(np.uint8)
	weights_pad = cv2.resize(weights_pad, (400, 30))
	return weights, weights_pad


TIME = 3000
MAX_STEP = 100000
MAX_EP = 100
SHOW_GRADS = False
SAVE_FIGS = False

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

instr_net = load_networks(nav_path='/home/andromeda/PycharmProjects/CustomUnreal/My_Agent/models/model_instruction_FINAL_4.pt', train=SHOW_GRADS) #model_nav_zero_fov.pt train=SHOW_GRADS) #GAU_2 (3 cancellato) BEST

tokenizer = Tokenizer()
instruction_vocabulary = ['go', 'straight', 'at', 'the', 'next', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'intersections', 'intersection', 'turn', 'right', 'left']
tokenizer.fit_on_texts(instruction_vocabulary)

env = Environment(9733, tokenizer=tokenizer, no_env=False, test=True, image=False, render=True, toy=False, neg_rew=True, att_loss=True)

ep_reward = 0
reward = 0
h, h_dir = init_hidden()
obs, depth, instruction, prev_att = env.reset()
X = obs.shape[2]
Y = obs.shape[1]
att = prev_att

print(instruction)
instruction_embedding = instr_net.get_embedding(instruction.cuda(), 1).detach()

cam = torch.zeros((1, 9, 9))

obs, frame, frame_cam, frame_norm = update_image(obs, cam)

pl1 = plt.subplot(111)
x = np.linspace(0, 200, 200)
val = np.zeros_like(x)
img_val, = plt.plot(x, val)

start = time.time()

#count = 10383
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('test10.avi',fourcc, 25.0, (224,224),1)
#out1 = cv2.VideoWriter('test10cam.avi',fourcc, 25.0, (224,224),1)
#with open('/home/lince/PycharmProjects/CustomUnreal/My_Agent/IMPALA_NAV/value.csv', mode='a') as file:
#file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
step = 0
ep = 0
neg = 0
while ((time.time() - start) < TIME) and (reward != 1) and step < MAX_STEP and ep < MAX_EP:
	step += 1
	#obs = obs if instruction[prev_att] == 2 else obs ** (instruction[att] - 18).type(torch.FloatTensor)
	#instr_net.choose_action(obs.cuda(), h, h_dir, instruction_embedding, torch.ones((1)).cuda())
	#action, h, h_dir, logits, v, _ = instr_net.choose_action(obs.cuda(), h, h_dir, instruction_embedding, torch.argmax(instr_net.direct, -1).type(torch.FloatTensor).cuda() + 1, train=SHOW_GRADS)
	#action, h, h_dir, logits, v, _ = instr_net.choose_action(obs.cuda(), h, h_dir, instruction_embedding, torch.ones((1)).type(torch.FloatTensor).cuda() if instruction[torch.argmax(instr_net.direct, -1)] == 2 else (instruction[torch.argmax(instr_net.direct, -1)] - 18).type(torch.FloatTensor).cuda(), train=SHOW_GRADS)
	#action, h, h_dir, logits, v, _ = instr_net.choose_action(obs.cuda(), h, h_dir, instruction_embedding, None, train=SHOW_GRADS)
	action, h, h_dir, logits, v = instr_net.choose_action(obs.cuda(), h, h_dir, instruction_embedding, None, train=SHOW_GRADS)
	#print(instruction_vocabulary[torch.argmax(instr_net.direct, -1) + 18], instruction_vocabulary[instruction[prev_att] - 1])
	#print(instruction_vocabulary[instruction[torch.argmax(instr_net.direct, -1)] - 1], instruction_vocabulary[instruction[prev_att] - 1])
	#cam = get_gradients(instr_net, logits[action], x2)
	#att_weights, weights_pad = get_att_frame(instr_net.net_dir, instruction)

	#action = torch.randint(0, 3, (1,))
	#print(action)

	#print(step, action.item())
	#action = torch.randint(0, 4, (1,))
	#action = 2
	reward, obs_, depth_, att = env.env_step(action)
	ep_reward += reward
	if prev_att != att:
		prev_att = att
		#print(instruction_vocabulary[torch.argmax(instr_net.direct, -1) + 18], instruction_vocabulary[instruction[prev_att] - 1])
		#time.sleep(2)
	#print(action, reward)

	#if reward != 0:
	#	print(reward)

	#print(value_input)

	if reward == 1 or step % 900 == 0 or reward < 0:	 # or reward < 0
		ep += 1
		obs_, depth_, instruction, prev_att = env.reset()
		print([instruction_vocabulary[i - 1] for i in instruction])
		#print(instruction_vocabulary[instruction[prev_att] - 1], prev_att)
		instruction_embedding = instr_net.get_embedding(instruction.cuda(), 1).detach()
		#if step % 900 == 0:
		h, h_dir = init_hidden()
		if reward < 0 or step % 900 == 0:
			neg += 1
		#else:
		#	print(step)
		step = 0
		#print('#####################################################################')

	reward = 0

	obs = obs_.cpu()
	depth = depth_

	obs, frame, frame_cam, frame_norm = update_image(obs, cam)

	val = update_val(val, v)
	pl1.set_ylim([val.min(), val.max()])
	img_val.set_ydata(val)

	cv2.imshow('frame', frame)
	#cv2.imshow('att', att_weights)
	#cv2.imshow('att_pad', weights_pad)
	#cv2.imshow('frame_cam', frame_cam)

	#out.write(frame)
	#out1.write(frame_cam)
	#cv2.imshow('frame_norm', frame_norm)

	#print(step, instruction_vocabulary[torch.argmax(instr_net.direct, -1) + 18])
	#	cv2.imwrite('/home/andromeda/PycharmProjects/CustomUnreal/My_Agent/instr_frames/img_'+str(step)+'.png', frame) 	TO TAKE SCREENSHOTS
	# np.save('img/features_'+str(count), x3.data.cpu().numpy())
	# file_writer.writerow(['features_'+str(count)+'.npy', v.item(), action.item()])
	# count = count+1

	if SAVE_FIGS and step > 200:
		#cv2.imshow('depth', (depth.view(40, 80).numpy() * 255).astype(np.uint8))
		cv2.imwrite('/home/andromeda/PycharmProjects/CustomUnreal/My_Agent/frames/img_' + str(step) + '.jpg', frame)
		#cv2.imwrite('/home/andromeda/PycharmProjects/CustomUnreal/My_Agent/frames/cam_' + str(step) + '.jpg', frame_cam)
		#cv2.imwrite('depth.jpg', (depth.view(40, 80).numpy() * 255).astype(np.uint8))
		plt.savefig('/home/andromeda/PycharmProjects/CustomUnreal/My_Agent/frames/val_' + str(step) + '.jpg', bbox_inches='tight')

	cv2.waitKey(1)
	#reward = 0
	#plt.pause(0.00001)

#print(count)
#print(reward)
#print(time.time() - start)
#print(step)
print(neg)
print(ep)
env.close_connection()
cv2.destroyAllWindows()
