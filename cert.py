import scipy.special
from scipy.stats import norm, binom_test,sem
from statsmodels.stats.proportion import proportion_confint
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from util import build_dirs


parser = argparse.ArgumentParser(description='Certification')
parser.add_argument('--exp_name', type=str, default="cert_dqn",
                    help='the name of the experiment to read data from')
parser.add_argument('--env_id', type=str, default="cartpole_simple", 
                    choices=["cartpole_simple", "cartpole_multiframe", "pong1r", "freeway", "bankheist", "highway"],
                    help='the id of the gym environment to read data from')
parser.add_argument('--num_models', type=int, default=1,
                     help='Number of models in the ensemble used to produce the data.') 
parser.add_argument('--lamda', type=str, default="1.0",
                     help='CAMP lamda value to certify')
parser.add_argument('--to_plot', type=str, choices=["comparison", "ablation"],
                    help='the experiment to plot')
args = parser.parse_args()

# For full results in Figure 8:
# colors = ['black', 'blue', 'green', 'orange', 'yellow', 'magenta']
# linestyles = [(0, (1, 1)), (5, (10, 3)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]

# For results in Figure 2:
colors = ['black', 'green', 'yellow'] # For Cartpole and Highway in Figure 2
#colors = ['green', 'blue'] # For Atari in Figure 2
linestyles = [(0, (1, 1)), (0, (5, 10)), (0, (5, 1))]


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

def compute_one_minus_cdf_function(sorted_values, lb_func):
	last_value = 0
	total = 0
	for idx,v in enumerate(sorted_values):
		if (v == last_value):
			continue
		else:
			one_minus_cdf = (len(sorted_values) - idx)/float(len(sorted_values)) # 1-cdf for the interval between x_(i-1) and x_i
			lb_one_minus_cdf = lb_func(one_minus_cdf)
			total += lb_one_minus_cdf * (v-last_value) 
			last_value = v
	#print(total)
	return total


def dkw_cohen(one_minus_cdf_empirical, n, alpha, eps,sigma):
	dkw = max(one_minus_cdf_empirical - math.sqrt(math.log(2/alpha)/(2*n)), 0)
	return get_exact_total(dkw,eps,sigma)


def cartpole_certify(data, sigma):
    data = torch.tensor(data)
    probs = torch.tensor(list([(data > i).sum()/10000. for i in range(200)]))
    probs_lb = [ _lower_confidence_bound(int(i*10000.),10000, .05/200) for i in probs]
    print(probs)
    print(probs_lb)
    vals = []
    for eps in np.arange(0.01,1.01, 0.01):
        avg = 0
        accum = 0
        for i in range(200):
            avg += probs[(199-i)]
            accum += get_exact_total(probs_lb[(199-i)],eps, sigma)
        if (eps == 0.01):
            vals.append(avg)
        vals.append(accum)
    return vals

def highway_certify(data, sigma):
    sorted_values = list([sum(x)  for x in data])
    sorted_values.sort()
    sorted_values = torch.tensor(sorted_values,device='cuda')
    vals = []
    for eps in np.arange(0, 2.01 , 0.01):
        vals.append(compute_one_minus_cdf_function(sorted_values, lambda x : dkw_cohen(x, 10000, 0.05, eps, sigma)).cpu())
    return vals

def freeway_certify(data, sigma):
    sorted_values = list([sum(x)  for x in data])
    sorted_values.sort()
    sorted_values = torch.tensor(sorted_values,device='cuda')
    vals = []
    for eps in np.arange(0,0.41, 0.01):
        vals.append(compute_one_minus_cdf_function(sorted_values, lambda x : dkw_cohen(x, 10000, 0.05, eps, sigma)).cpu())
    return vals


def pong_certify(data, sigma):
    prob = (torch.tensor([x[0] for x in data])/2.+.5).mean()
    lcb = _lower_confidence_bound(int(prob*10000), 10000, 0.05)
    vals = [lcb]
    for eps in np.arange(0.01,0.41, 0.01):
        vals.append(get_exact_total(lcb,eps,sigma))
    return vals


def plot_cert_comparison():
    plt.figure(figsize=(8.4,4.8))
    if args.env_id == "cartpole_simple":
        title = 'Cartpole-1'
    elif args.env_id == "cartpole_multiframe":
        title = 'Cartpole-5'
    elif args.env_id == "freeway":
        title = 'Freeway'
    elif args.env_id == "pong1r":
        title = 'Pong (Single Round)'
    elif args.env_id == "highway":
        title = 'Highway'
    elif args.env_id == "bankheist":
        title = 'Bank Heist'
    
    if "cartpole" in args.env_id:
        eps_range = 1.01
        #sigma_list = [0.2, 0.4, 0.6, 0.8, 1.0] # For full results in Figure 8
        #sigma_list = [0.2, 0.6, 1.0] # For results in Figure 2
        sigma_list = [0.2] # For USENIX Security'25 AE only
    elif "highway" in args.env_id:
        eps_range = 2.01
        #sigma_list = [0.4, 0.8, 1.2, 1.6, 2.0] # For full results in Figure 8
        sigma_list = [0.4, 1.2, 2.0] # For results in Figure 2
    else:
         eps_range = 0.41
         sigma_list = [0.05, 0.1]
    for j, sigma in enumerate(sigma_list):
        camp_path = f"./eval_exp/test_dqn_camp/{args.env_id}/wrapper_sigma={sigma}-lamda={args.lamda}-eval/eval_results.pth"
        gaussian_path = f"./eval_exp/test_dqn_baseline/{args.env_id}/wrapper_sigma={sigma}-eval/eval_results.pth"
        camp_data = torch.load(camp_path)
        baseline_data = torch.load(gaussian_path)
        if "cartpole" in args.env_id or "highway" in args.env_id:
            noisynet_path = f"./eval_exp/test_dqn_noisynet/{args.env_id}/wrapper_sigma={sigma}-eval/eval_results.pth"
            noisynet_data = torch.load(noisynet_path)
        
        if "cartpole" in args.env_id:
            camp_vals = cartpole_certify(camp_data, sigma)
            baseline_vals = cartpole_certify(baseline_data, sigma)
            noisynet_vals = cartpole_certify(noisynet_data, sigma)
        
        elif "highway" in args.env_id:
            camp_vals = highway_certify(camp_data, sigma)
            baseline_vals = highway_certify(baseline_data, sigma)
            noisynet_vals = highway_certify(noisynet_data, sigma)    
        
        elif "pong" in args.env_id:
            camp_vals = pong_certify(camp_data, sigma)
            baseline_vals = pong_certify(baseline_data, sigma)
        
        else:
            camp_vals = freeway_certify(camp_data, sigma)
            baseline_vals = freeway_certify(baseline_data, sigma)

        print("Gaussian:", baseline_vals)
        print("CAMP:", camp_vals)
        plt.plot(np.arange(0.00, eps_range, 0.01),
                camp_vals,
                color=colors[j],
                linestyle=linestyles[0],  
                label= "CAMP $\sigma$ = " + str(sigma))
        plt.plot(np.arange(0.00, eps_range, 0.01),
                baseline_vals, 
                color=colors[j],
                linestyle='-',  
                label= "Gaussian $\sigma$ = " + str(sigma))
        if "cartpole" in args.env_id or "highway" in args.env_id:
            plt.plot(np.arange(0.00, eps_range, 0.01),
                    noisynet_vals, 
                    color=colors[j],
                    linestyle='--',  
                    label= "NoisyNet $\sigma$ = " + str(sigma))
        plt.xlim(0, eps_range)
    
    plt.legend()
    plt.tight_layout()
    plt.title(title, fontsize=18)
    plt.xlabel('Perturbation Budget', fontsize=16)
    if "cartpole" in args.env_id:
        plt.ylim(0, 201)
    elif "freeway" in args.env_id:
        plt.ylim(0, 5)
    elif "pong" in args.env_id:
        plt.ylim(0, 1)
    elif "bankheist" in args.env_id:
        plt.ylim(0, 201)
    elif "highway" in args.env_id:
        plt.ylim(0, 30)

    plt.ylabel('Certified Expected Return', fontsize=16)
    plt.savefig(f'exp_cert/comparisons/{args.env_id}-{args.exp_name}.pdf', dpi=400, bbox_inches='tight')
    return


def plot_cartpole_ablation():
    if args.env_id == "cartpole_simple":
        title = 'Cartpole-1'
    elif args.env_id == "cartpole_multiframe":
        title = 'Cartpole-5'
    for sigma in [0.2, 0.4, 0.6, 0.8, 1.0]:
        plt.figure(figsize=(5.04, 2.88))
        for j, la in enumerate([0.5, 1.0, 2.0, 4.0, 8.0, 16.0]):
            camp_path = f"./eval_exp/test_dqn_camp/{args.env_id}/wrapper_sigma={sigma}-lamda={la}-eval/eval_results.pth"
            camp_data = torch.tensor(torch.load(camp_path))
            camp_vals = cartpole_certify(camp_data, sigma)
            plt.plot(np.arange(0.00,1.01, 0.01),
                    camp_vals, 
                    color=colors[j],
                    linestyle=linestyles[j],
                    label= "CAMP $\lambda$ = " + str(la))
        plt.legend()
        plt.title(title, fontsize=18)
        plt.xlim(0, 1.)
        plt.xlabel('Perturbation Budget', fontsize=16)
        plt.ylim(0, 201)
        plt.ylabel('Certified Expected Return', fontsize=16)
        plt.savefig(f'exp_cert/ablations/{args.env_id}-{args.exp_name}-ablation-sigma={sigma}.pdf', dpi=400, bbox_inches='tight')
        plt.close()
    return

if __name__ == "__main__":
    build_dirs("exp_cert")
    build_dirs("exp_cert/comparisons")
    build_dirs("exp_cert/ablations")    
    sns.set_theme(style="darkgrid")
    if args.to_plot == "comparison":
        plot_cert_comparison()
    elif args.to_plot == "ablation":
        plot_cartpole_ablation()
    else:
        raise NotImplementedError
    
    