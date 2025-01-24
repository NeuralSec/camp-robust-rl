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
from apgdt_multiframe import APGDT

parser = argparse.ArgumentParser(description='PyTorch Highway attack')
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
					help='Initial attack step size for calculating step counts')
parser.add_argument('--attack_step_count_multiplier', type=float, default=2, metavar='N',
					help='Multiplier between steps and budget/attack_step')
args = parser.parse_args()


def attack_clean(state,model,carryover_budget_squared, clean_prev_obs_tens = None, dirty_prev_obs_tens = None):
	if (clean_prev_obs_tens is not None):
		clean_prev_obs_tens = torch.stack(clean_prev_obs_tens).reshape(-1).cuda().unsqueeze(0)
		dirty_prev_obs_tens = torch.cat(dirty_prev_obs_tens,dim=0).reshape(-1).cuda().unsqueeze(0)

	state = torch.tensor(state,device='cuda').float().unsqueeze(0)
	if (clean_prev_obs_tens is not None):
		target_logits =  model(torch.cat([clean_prev_obs_tens,state],dim=1))[0]
	else:
		target_logits =  model(state.repeat(1,5))[0]
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
	attack = APGDT(model, eps=budget, steps=step_count, n_classes=target_logits.shape[-1])
	for target in targets:
		attack.target_class = target
		state = copy.deepcopy(ori_state.data)
		best_state, best_q = attack.attack_single_run(state, best_state, best_q, target_logits, clean_prev_obs_tens, dirty_prev_obs_tens)
	return best_state[0].unsqueeze(0), budget**2 - (best_state - ori_state).norm()**2


from models import MLP


if __name__ == '__main__':
	args.store_all_rewards = False
	for arg in vars(args):
		print("%s: %s" % (arg, getattr(args, arg)))
	device='cuda'
	env =  gym.make("CartPole-v0")
	reward_accum = []
	policy_kwargs = {}

	# Define model
	state, _ = env.reset(seed=args.seed)
	act_dim = env.action_space.n
	stat_dim = len(state)*5
	dqn_ensemble = [MLP(stat_dim, act_dim).to(device)]

	# load checkpoint
	if args.checkpoint_path:
		checkpoint = torch.load(args.checkpoint_path)
		[dqn.load_state_dict(checkpoint['model_state_dict'][i]) for i, dqn in enumerate(dqn_ensemble)]

	[dqn.eval() for dqn in dqn_ensemble]
	model = dqn_ensemble[0]

	for i_episode in range(args.num_evals):
		state, _ = env.reset()
		ep_reward =0
		policy_rewards = []
		carryover = args.attack_eps*args.attack_eps
		frame_hist = []
		state_hist = []

		for t in range(1, 201): 
			#print(carryover)
			if (t == 1):
				observation,carryover = attack_clean(state,model,carryover)
				frame_hist.extend([observation]*5)
				state_hist.extend([torch.tensor(state)]*5)
			else:
				observation,carryover = attack_clean(state,model,carryover, clean_prev_obs_tens =state_hist[-4:],dirty_prev_obs_tens =frame_hist[-4:])
				frame_hist.append(observation)
				state_hist.append(torch.tensor(state))

			action = model(torch.stack(frame_hist[-5:]).reshape(-1).unsqueeze(0)).max(1).indices.view(1,1)
			state, reward, terminated, truncated, _ = env.step(action.item())
			done = terminated or truncated
			if args.render:
				env.render()
			policy_rewards.append(reward)
			ep_reward += reward
			if done:
				break
		if (args.store_all_rewards):
			reward_accum.append(policy_rewards)
		else:
			print(ep_reward)
			reward_accum.append(ep_reward)
		if i_episode % args.log_interval == 0:
			print('Episode {}\t'.format(
				i_episode),flush=True)
	torch.save(reward_accum, args.checkpoint_path + '_evals_'+ str(args.num_evals) + '_APGD_eps_' + str(args.attack_eps) + '_attack_step_count_multiplier_' + str(args.attack_step_count_multiplier) + '_threshold_'+ str(args.q_threshold)+ '.pth')
