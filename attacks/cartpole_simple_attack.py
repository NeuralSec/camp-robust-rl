import argparse
import gym
import numpy as np
from itertools import count
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch cartpole attack')
parser.add_argument('--seed', type=int, default=0, metavar='N',
					help='random seed (default: 0)')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--q_threshold', type=float, default=0., metavar='N',
					help='q value threshold')
parser.add_argument('--num_evals', type=int, default=1000, metavar='N',
					help='Number of evaluations')
parser.add_argument('--store_all_rewards', action='store_true',
					help='store all rewards (vs just sum)')
parser.add_argument('--checkpoint_path', type=str,
					help='checkpoint path')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--attack_eps', type=float, default=0.2, metavar='N',
					help='Attack epsilon, total')
parser.add_argument('--attack_step', type=float, default=0.01, metavar='N',
					help='Attack step size')
parser.add_argument('--attack_step_count_multiplier', type=float, default=2, metavar='N',
					help='Multiplier between steps and budget/attack_step')
args = parser.parse_args()


def attack_clean(state,model,carryover_budget_squared):
	state = torch.tensor(state,device='cuda').float().unsqueeze(0)
	target_logits =  model(state)[0]
	starting_action = target_logits.argmax()
	ori_state = copy.deepcopy(state.data)
	state= state.detach()
	obj = torch.nn.CrossEntropyLoss()
	if (carryover_budget_squared <= 0):
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
			out = model(state)
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
			state= state.detach_()

	return best_state[0].unsqueeze(0), budget**2 - (best_state - ori_state).norm()**2

from models import MLP


if __name__ == '__main__':
	args.store_all_rewards = False
	for arg in vars(args):
		print("%s: %s" % (arg, getattr(args, arg)))
	device='cuda'
	env =  gym.make("CartPole-v0")
	reward_accum = []
	state_hist = []
	policy_kwargs = {}

	# Define model
	state, _ = env.reset(seed=args.seed)
	act_dim = env.action_space.n
	stat_dim = len(state)
	dqn_ensemble = [MLP(stat_dim, act_dim).to(device)]

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
		state_hist.append(state)
		carryover = args.attack_eps*args.attack_eps
		for t in range(1, 201):
			obs, carryover = attack_clean(state,model,carryover)
			action = model(obs).max(1).indices.view(1,1)
			state, reward, terminated, truncated, _ = env.step(action.item())
			done = terminated or truncated
			state_hist.append(state)
			if args.render:
				env.render()
			policy_rewards.append(reward)
			ep_reward += reward
			if done:
				break
		if (args.store_all_rewards):
			reward_accum.append(policy_rewards)
		else:
			print("ep_reward:", ep_reward)
			reward_accum.append(ep_reward)
		if i_episode % args.log_interval == 0:
			print('Episode {}\t'.format(
				i_episode))
	print("Mean return:", np.mean(reward_accum))
	torch.save(reward_accum, args.checkpoint_path + '_evals_'+ str(args.num_evals) + '_attack_eps_' + str(args.attack_eps) + '_attack_step_count_multiplier_' + str(args.attack_step_count_multiplier) + '_attack_step_'+ str(args.attack_step)+ '_threshold_'+ str(args.q_threshold)+ '.pth')
