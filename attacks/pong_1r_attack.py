import argparse
import gymnasium as gym
import numpy as np
from itertools import count
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.distributions import Categorical

import time
parser = argparse.ArgumentParser(description='PyTorch pong attack')
parser.add_argument('--seed', type=int, default=0, metavar='N',
					help='random seed (default: 0)')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--num_evals', type=int, default=1000, metavar='N',
					help='Number of evaluations')
parser.add_argument('--store_all_rewards', action='store_true',
					help='store all rewards (vs just sum)')
parser.add_argument('--checkpoint_path', type=str,
					help='checkpoint path')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--attack_eps', type=float, default=.1, metavar='N',
					help='Attack epsilon, total')
parser.add_argument('--q_threshold', type=float, default=.3, metavar='N',
					help='q value threshold')
parser.add_argument('--attack_step', type=float, default=0.01, metavar='N',
					help='Attack step size')
parser.add_argument('--attack_step_count_multiplier', type=float, default=2, metavar='N',
					help='Multiplier between steps and budget/attack_step')
args = parser.parse_args()

class FinishEarlyWrapper(gym.Wrapper):
	def reset(self, **kwargs):
		return self.env.reset(**kwargs)

	def step(self, action):
		observation, reward, terminated, truncated, info = self.env.step(action)
		if (reward !=0):
			terminated = True
		return observation, reward, terminated, truncated, info

class NoisyObsWrapper(gym.ObservationWrapper):
	def __init__(self, env, sigma):
		super().__init__(env)
		self.sigma = sigma
	def observation(self, obs):
		return obs + self.sigma*np.random.standard_normal(size=obs.shape)
	
def attack_clean(state,model,carryover_budget_squared, clean_prev_obs_tens = None, dirty_prev_obs_tens = None):
	if (clean_prev_obs_tens is not None):
		clean_prev_obs_tens = torch.stack(clean_prev_obs_tens).cuda().unsqueeze(0)
		dirty_prev_obs_tens = torch.cat(dirty_prev_obs_tens,dim=0).cuda().unsqueeze(0)

	state = torch.tensor(state,device='cuda').float().unsqueeze(0).unsqueeze(0)
	if (clean_prev_obs_tens is not None):
		target_logits =  model(torch.cat([clean_prev_obs_tens,state],dim=1))[0]
	else:
		target_logits =  model(state.repeat(1,4,1,1))[0]
	starting_action = target_logits.argmax()
	ori_state = copy.deepcopy(state.data)
	state= state.detach()
	obj = torch.nn.CrossEntropyLoss()
	if (carryover_budget_squared <= 0) or math.isnan(math.sqrt(carryover_budget_squared)):
		budget = 0
	else:
		budget = math.sqrt(carryover_budget_squared)
	step_count = int(args.attack_step_count_multiplier * budget/args.attack_step)
	targets = list([torch.tensor(x).cuda() for x in range(target_logits.shape[0])])
	[targets.remove(x.item()) for x in (target_logits + args.q_threshold >target_logits[starting_action]).nonzero(as_tuple=False)]
	best_q = target_logits[starting_action]
	best_state = ori_state
	for target in targets:
		state =  copy.deepcopy(ori_state.data)
		for i in range(step_count):
			state.requires_grad = True
			if (clean_prev_obs_tens is not None):
				out = model(torch.cat([dirty_prev_obs_tens,state],dim=1))
			else:
				out = model(state.repeat(1,4,1,1))
			if (out[0].argmax() == target):
				if target_logits[target] < best_q:
					best_q = target_logits[target]
					best_state = state.detach_()
				break
			model.zero_grad()
			cost = -obj(out,target.unsqueeze(0))
			grad, = torch.autograd.grad(inputs=state, outputs=cost)
			state = state + args.attack_step*grad/grad.norm()
			if ((state - ori_state).norm() > budget):
				state = ori_state + (state - ori_state) * budget / (state - ori_state).norm()
			state.clamp_(0,1)
			state= state.detach_()
	return best_state[0], budget**2 - (best_state - ori_state).norm()**2

from models import NatureCNN
from gymnasium.wrappers import FrameStack, AtariPreprocessing

if __name__ == '__main__':
	args.store_all_rewards = True
	for arg in vars(args):
		print("%s: %s" % (arg, getattr(args, arg)))
	device='cuda'
	env =  FinishEarlyWrapper(AtariPreprocessing(gym.make("PongNoFrameskip-v0"), scale_obs=True))
	reward_accum = []
	policy_kwargs = {}

	state, _ = env.reset(seed=args.seed)
	act_dim = env.action_space.n
	dqn_ensemble = [NatureCNN(frames=4, act_dim=act_dim).to(device)]
	
	# load checkpoint
	if args.checkpoint_path:
		checkpoint = torch.load(args.checkpoint_path)
		[dqn.load_state_dict(checkpoint['model_state_dict'][i]) for i, dqn in enumerate(dqn_ensemble)]

	[dqn.eval() for dqn in dqn_ensemble]
	model = dqn_ensemble[0]

	# Seeding
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	env.action_space.seed(args.seed)
	env.observation_space.seed(args.seed)

	for i_episode in range(args.num_evals):
		state, _ = env.reset()
		ep_reward =0
		policy_rewards = []
		state_hist = []
		carryover = args.attack_eps*args.attack_eps
		frame_hist = []
		t = 0
		after_first_loss = False
		while True: 
			t += 1
			if (t == 1):
				observation,carryover = attack_clean(state,model,carryover)
				frame_hist.extend([observation]*4)
				state_hist.extend([torch.tensor(state,device='cuda')]*4)
			else:
				observation,carryover = attack_clean(state,model,carryover, clean_prev_obs_tens =state_hist[-3:],dirty_prev_obs_tens =frame_hist[-3:])
				frame_hist.append(observation)
				state_hist.append(torch.tensor(state,device='cuda'))
			action = model(torch.cat(frame_hist[-4:],dim=0).unsqueeze(0)).max(1).indices.view(1,1)
			state, reward, terminated, truncated, _ = env.step(action.item())
			done = terminated or truncated
			if args.render:
				env.render()
			if (reward != 0):
				policy_rewards.append(reward)
			ep_reward += reward
			if done:
				break
		if (args.store_all_rewards):
			reward_accum.append(policy_rewards)
		else:
			reward_accum.append(ep_reward)
		if i_episode % args.log_interval == 0:
			print('Episode '+ str(i_episode), flush=True)
	torch.save(reward_accum, args.checkpoint_path + '_evals_'+ str(args.num_evals) + '_attack_eps_' + str(args.attack_eps) + '_attack_step_count_multiplier_' + str(args.attack_step_count_multiplier) + '_attack_step_'+ str(args.attack_step)+ '_threshold_'+ str(args.q_threshold) +'.pth')
