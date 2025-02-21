
import scipy.special
from scipy.stats import norm, binom_test,sem
from statsmodels.stats.proportion import proportion_confint
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='PyTorch Highway attack')
parser.add_argument('--attack_type', type=str, default="pgd",
                    choices=["pgd", "apgd"],
                    help='select attack method')
parser.add_argument('--num_evals', type=int, default=1000,
					help='Number of evaluations')
args = parser.parse_args()

env_id="cartpole_simple"
q_gap = 0.0
lamda=1.0

def get_cvar_cert_time_t(estimate, t, eps,sigma):
	erf = scipy.special.erf(math.sqrt(t+1) * eps/(2*math.sqrt(2)*sigma))
	cvar = 1. if estimate > erf else estimate/erf
	return cvar * erf

def get_exact_time_t(estimate, t, eps,sigma):
	return norm.cdf(norm.ppf(estimate) - math.sqrt(t+1) * eps/(sigma))

def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
def get_exact_total(estimate, eps,sigma):
	return norm.cdf(norm.ppf(estimate) -  eps/(sigma))

def _hoeffding_lcb(mean: float,  N: int, alpha: float) -> float:
	return max(mean - math.sqrt(math.log(1./alpha)/(2*N)),0)


sns.set_theme(style="darkgrid")
plt.figure(figsize=(8.4,4.8))
linestyles = [(0, (1, 1)), (5, (10, 3)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
attack_mags_nonzero = [0.2, 0.4, 0.6, 0.8, 1.0]
#sigma_list = [0.0, 0.2, 0.6] # Use this to produce Figures 3 and 4.
#colors = ['purple', 'black','green'] # Use this to produce Figures 3 and 4.
sigma_list = [0.2] # For USENIX artifact evaluation
colors = ['black'] # For USENIX artifact evaluation

def plot_gaussian():
	# Without defense + with Gaussian
	for j,sigma in enumerate(sigma_list):
		path = f'../eval_exp/test_dqn_baseline/{env_id}/wrapper_sigma={sigma}-eval/eval_results.pth'
		attack_vals =  [torch.tensor(torch.load(path)).float().mean().item()]
		attack_sems = [sem(torch.tensor(torch.load(path)))]
		for attack_mag in attack_mags_nonzero:
			attack_val = None
			attack_sem = None
			for i,thresh in enumerate([q_gap]):
				if args.attack_type == "pgd":
					data_path=f"../exp/train_dqn_baseline/{env_id}/wrapper_sigma={sigma}-seed=0/checkpoints/{env_id}_restore.pth_" + \
						f'evals_{args.num_evals}_attack_eps_' + str(attack_mag) + '_attack_step_count_multiplier_2' + '_attack_step_0.01_threshold_'+ str(thresh)+ '.pth'
				elif args.attack_type == "apgd":
					data_path=f"../exp/train_dqn_baseline/{env_id}/wrapper_sigma={sigma}-seed=0/checkpoints/{env_id}_restore.pth_" + \
						f'evals_{args.num_evals}_APGD_eps_' + str(attack_mag) + '_attack_step_count_multiplier_2' + '_threshold_'+ str(thresh)+ '.pth'
				cur_val = (torch.tensor( torch.load(data_path)).float().mean().item())
				if (attack_val is None or cur_val < attack_val):
					attack_val = cur_val
					attack_sem = (sem(torch.tensor( torch.load(data_path))))
			print(f"Gaussian: Sigma={sigma}, current eps is: {attack_mag} and average return is {attack_val}")
			attack_vals.append(attack_val)
			attack_sems.append(attack_sem)
		attack_mags = [0] + attack_mags_nonzero
		print(f"Gaussian: Ploting a curve: at sigma={sigma}, current eps are: {attack_mags} and average return are {attack_vals}")
		plt.errorbar([x for x in attack_mags], attack_vals, yerr= attack_sems, 
				color=colors[j], linestyle ="-", label="Gaussian (σ = " + str(sigma) + ')')
	return


def plot_camp():
	# CAMP
	for j,sigma in enumerate(sigma_list):
		path = f'../eval_exp/test_dqn_camp/{env_id}/wrapper_sigma={sigma}-lamda={lamda}-eval/eval_results.pth'
		attack_vals =  [torch.tensor(torch.load(path)).float().mean().item()]
		attack_sems = [sem(torch.tensor(torch.load(path)))]
		for attack_mag in attack_mags_nonzero:
			attack_val = None
			attack_sem = None
			for i,thresh in enumerate([q_gap]):
				if args.attack_type == "pgd":
					data_path=f"../exp/train_dqn_camp/{env_id}/wrapper_sigma={sigma}-lamda={lamda}-seed=0/checkpoints/{env_id}_restore.pth_" + \
						f'evals_{args.num_evals}_attack_eps_' + str(attack_mag) + '_attack_step_count_multiplier_2' + '_attack_step_0.01_threshold_'+ str(thresh)+ '.pth'
				elif args.attack_type == "apgd":
					data_path=f"../exp/train_dqn_camp/{env_id}/wrapper_sigma={sigma}-lamda={lamda}-seed=0/checkpoints/{env_id}_restore.pth_" + \
						f'evals_{args.num_evals}_APGD_eps_' + str(attack_mag) + '_attack_step_count_multiplier_2' + '_threshold_'+ str(thresh)+ '.pth'
				cur_val = (torch.tensor(torch.load(data_path)).float().mean().item())
				if (attack_val is None or cur_val < attack_val):
					attack_val = cur_val
					attack_sem = (sem(torch.tensor(torch.load(data_path))))
			print(f"CAMP: Sigma={sigma}, current eps is: {attack_mag} and average return is {attack_val}")
			attack_vals.append(attack_val)
			attack_sems.append(attack_sem)
		attack_mags = [0] + attack_mags_nonzero
		print(f"CAMP: Ploting a curve: at sigma={sigma}, current eps are: {attack_mags} and average return are {attack_vals}")
		plt.errorbar([x for x in attack_mags],attack_vals,  yerr= attack_sems, 
				color=colors[j], linestyle =linestyles[0], label="CAMP (σ = " + str(sigma) + ')')
	return

plot_gaussian()
plot_camp()
plt.legend()
plt.title('Cartpole-1', fontsize=18)
plt.xlim(0,1.)
plt.xlabel('Perturbation Budget', fontsize=14)
plt.ylim(0,201)
plt.ylabel('Average Return', fontsize=14)
plt.savefig(f'cartpole_simple_{args.attack_type}_threshold={q_gap}.pdf', dpi=400,bbox_inches='tight')
