
import scipy.special
from scipy.stats import norm, binom_test,sem
from statsmodels.stats.proportion import proportion_confint
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


env_id="bankheist"
attack_type="apgd" # select from {"pgd", "apgd"}
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
import argparse



sns.set_theme(style="darkgrid")
plt.figure(figsize=(8.4,4.8))
colors = ['green', 'blue']
linestyles = [(0, (1, 1)), (5, (10, 3)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
attack_mags_nonzero = [0.05, 0.15, 0.25, 0.35]


def plot_gaussian():
    for j,sigma in enumerate([0.05, 0.1]):
        path = f'../eval_exp/test_dqn_baseline/{env_id}/wrapper_sigma={sigma}-eval/eval_results.pth'
        attack_vals = [torch.tensor([sum(x) for x in torch.load(path)]).mean().item()]
        attack_sems = [sem(torch.tensor([sum(x) for x in torch.load(path)]))]
        for attack_mag in attack_mags_nonzero:
            attack_val = None
            attack_sem = None
            for i,thresh in enumerate([0.0]):
                if attack_type == "pgd":
                    data_path=f"../exp/train_dqn_baseline/{env_id}/wrapper_sigma={sigma}-seed=0/checkpoints/{env_id}_restore.pth_" + \
                        'evals_1000_attack_eps_' + str(attack_mag) + '_attack_step_count_multiplier_2' + '_attack_step_0.01_threshold_'+ str(thresh)+ '.pth'
                elif attack_type == "apgd":
                    data_path=f"../exp/train_dqn_baseline/{env_id}/wrapper_sigma={sigma}-seed=0/checkpoints/{env_id}_restore.pth_" + \
						'evals_1000_APGD_eps_' + str(attack_mag) + '_attack_step_count_multiplier_2' + '_threshold_'+ str(thresh)+ '.pth'
                cur_val = (torch.tensor( [sum(x) for x in torch.load(data_path)]).mean().item())
                if (attack_val is None or cur_val < attack_val):
                    attack_val = cur_val
                    attack_sem = (sem(torch.tensor( [sum(x) for x in  torch.load(data_path)])))
            attack_vals.append(attack_val)
            attack_sems.append(attack_sem)
        attack_mags = [0] + attack_mags_nonzero
        print(attack_vals)
        plt.errorbar([x for x in attack_mags],attack_vals,  yerr= attack_sems, 
                color=colors[j], linestyle ="-", label="Gaussian (σ = " + str(sigma) + ')')
    return

def plot_camp():
    for j,sigma in enumerate([0.05, 0.1]):
        path = f'../eval_exp/test_dqn_camp/{env_id}/wrapper_sigma={sigma}-lamda={lamda}-eval/eval_results.pth'
        attack_vals = [torch.tensor([sum(x) for x in torch.load(path)]).mean().item()]
        attack_sems = [sem(torch.tensor([sum(x) for x in torch.load(path)]))]
        for attack_mag in attack_mags_nonzero:
            attack_val = None
            attack_sem = None
            for i,thresh in enumerate([0.0]):
                if attack_type == "pgd":
                    data_path=f"../exp/train_dqn_camp/{env_id}/wrapper_sigma={sigma}-lamda={lamda}-seed=0/checkpoints/{env_id}_restore.pth_" + \
                        'evals_1000_attack_eps_' + str(attack_mag) + '_attack_step_count_multiplier_2' + '_attack_step_0.01_threshold_'+ str(thresh)+ '.pth'
                elif attack_type == "apgd":
                    data_path=f"../exp/train_dqn_camp/{env_id}/wrapper_sigma={sigma}-lamda={lamda}-seed=0/checkpoints/{env_id}_restore.pth_" + \
						'evals_1000_APGD_eps_' + str(attack_mag) + '_attack_step_count_multiplier_2' + '_threshold_'+ str(thresh)+ '.pth'
                cur_val = (torch.tensor( [sum(x) for x in torch.load(data_path)]).mean().item())
                if (attack_val is None or cur_val < attack_val):
                    attack_val = cur_val
                    attack_sem = (sem(torch.tensor( [sum(x) for x in  torch.load(data_path)])))
            attack_vals.append(attack_val)
            attack_sems.append(attack_sem)
        attack_mags = [0] + attack_mags_nonzero
        plt.errorbar([x for x in attack_mags],attack_vals,  yerr= attack_sems, 
                color=colors[j], linestyle =linestyles[0], label="CAMP (σ = " + str(sigma) + ')')
    return

plot_gaussian()
plot_camp()
plt.legend()
plt.title('Bank Heist', fontsize=18)
plt.xlim(0, .36)
plt.xlabel('Perturbation Budget', fontsize=14)
plt.ylim(0, 200)
plt.ylabel('Average Return', fontsize=14)
plt.savefig(f'bankheist_{attack_type}_threshold={q_gap}.pdf',dpi=400,bbox_inches='tight')
