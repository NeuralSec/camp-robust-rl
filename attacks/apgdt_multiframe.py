import time

import numpy as np

import torch

from torchattacks.attack import Attack


class APGDT(Attack):
	r"""
	APGD-Targeted in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks.'
	Targeted attack for every wrong classes.
	[https://arxiv.org/abs/2003.01690]
	[https://github.com/fra31/auto-attack]

	Distance Measure : Linf, L2

	Arguments:
		model (nn.Module): model to attack.
		norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
		eps (float): maximum perturbation. (Default: 8/255)
		steps (int): number of steps. (Default: 10)
		n_restarts (int): number of random restarts. (Default: 1)
		seed (int): random seed for the starting point. (Default: 0)
		eot_iter (int): number of iteration for EOT. (Default: 1)
		rho (float): parameter for step-size update (Default: 0.75)
		verbose (bool): print progress. (Default: False)
		n_classes (int): number of classes. (Default: 10)

	Shape:
		- images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
		- labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
		- output: :math:`(N, C, H, W)`.

	Examples::
		>>> attack = torchattacks.APGDT(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, eot_iter=1, rho=.75, verbose=False, n_classes=10)
		>>> adv_images = attack(images, labels)

	"""

	def __init__(
		self,
		model,
		norm="L2",
		eps=8 / 255,
		steps=10,
		n_restarts=1,
		seed=0,
		eot_iter=1,
		rho=0.75,
		verbose=False,
		n_classes=10,
	):
		super().__init__("APGDT", model)
		self.eps = eps
		self.steps=steps
		self.norm = norm
		self.n_restarts = n_restarts
		self.seed = seed
		self.eot_iter = eot_iter
		self.thr_decr = rho
		self.verbose = verbose
		self.target_class = None
		self.n_target_classes = n_classes - 1
		self.supported_mode = ["default"]

	def check_oscillation(self, x, j, k, y5, k3=0.75):
		t = np.zeros(x.shape[1])
		for counter5 in range(k):
			t += x[j - counter5] > x[j - counter5 - 1]

		return t <= k * k3 * np.ones(t.shape)

	def check_shape(self, x):
		return x if len(x.shape) > 0 else np.expand_dims(x, 0)

	def attack_single_run(self, x, best_state, best_q, target_logits, clean_prev_obs_tens = None, dirty_prev_obs_tens = None):
		self.steps_2, self.steps_min, self.size_decr = (
			max(int(0.22 * self.steps), 1),
			max(int(0.06 * self.steps), 1),
			max(int(0.03 * self.steps), 1),
		)
		if self.verbose:
			print(
				"parameters: ", self.steps, self.steps_2, self.steps_min, self.size_decr
			)
		
		x_adv = x.detach() 
		x_best = x_adv.clone()
		x_best_adv = x_adv.clone()
		loss_steps = torch.zeros([self.steps, x.shape[0]])
		loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
		acc_steps = torch.zeros_like(loss_best_steps)
		
		x_adv.requires_grad_()
		grad = torch.zeros_like(x)
		obj = torch.nn.CrossEntropyLoss(reduction='none')
		for _ in range(self.eot_iter):
			with torch.enable_grad():
				if (clean_prev_obs_tens is not None):	
					logits = self.get_logits(torch.cat([dirty_prev_obs_tens,x_adv],dim=1))
				else:
					logits = self.get_logits(x_adv.repeat(1,5))
				loss_indiv = -obj(logits, self.target_class.unsqueeze(0))
				loss = loss_indiv.sum()

			grad += torch.autograd.grad(loss, [x_adv])[0].detach()

		grad /= float(self.eot_iter)
		grad_best = grad.clone()

		loss_best = loss_indiv.detach().clone()

		step_size = (
			self.eps
			* torch.ones([x.shape[0], 1]).to(self.device).detach()
			* torch.Tensor([0.01]).to(self.device).detach().reshape([1, 1])
		)
		x_adv_old = x_adv.clone()

		k = self.steps_2 + 0
		u = np.arange(x.shape[0])
		counter3 = 0

		loss_best_last_check = loss_best.clone()
		reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
		
		for i in range(self.steps):
			# Check step: check if target action is achieved and if the induced Q value is the current lowerest
			with torch.no_grad():
				if (logits[0].argmax() == self.target_class):
					if target_logits[self.target_class] < best_q:
						best_q = target_logits[self.target_class]
						best_state = x_adv.detach_()
						b_used = (best_state - x).norm()**2
					break

			# gradient step
			with torch.no_grad():
				x_adv = x_adv.detach()
				grad2 = x_adv - x_adv_old
				x_adv_old = x_adv.clone()

				a = 0.75 if i > 0 else 1.0

				if self.norm == "L2":
					x_adv_1 = x_adv + step_size[0] * grad / (
						(grad ** 2).sum(dim=(-1), keepdim=True).sqrt() + 1e-12
					)
					x_adv_1 = (
						x
						+ (x_adv_1 - x)
						/ (
							((x_adv_1 - x) ** 2).sum(dim=(-1), keepdim=True).sqrt()
							+ 1e-12
						)
						* torch.min(
							self.eps * torch.ones(x.shape).to(self.device).detach(),
							((x_adv_1 - x) ** 2)
							.sum(dim=(-1), keepdim=True)
							.sqrt(),
						)
					)
					x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
					x_adv_1 = (
						x
						+ (x_adv_1 - x)
						/ (
							((x_adv_1 - x) ** 2).sum(dim=(-1), keepdim=True).sqrt()
							+ 1e-12
						)
						* torch.min(
							self.eps * torch.ones(x.shape).to(self.device).detach(),
							((x_adv_1 - x) ** 2).sum(dim=(-1), keepdim=True).sqrt()
							+ 1e-12,
						)
					)
				
				else:
					raise NotImplementedError
				
				x_adv = x_adv_1 + 0.0
			
			# get gradient
			x_adv.requires_grad_()
			grad = torch.zeros_like(x)
			for _ in range(self.eot_iter):
				with torch.enable_grad():
					if (clean_prev_obs_tens is not None):	
						logits = self.get_logits(torch.cat([dirty_prev_obs_tens,x_adv],dim=1))
					else:
						logits = self.get_logits(x_adv.repeat(1,5))
					loss_indiv = -obj(logits, self.target_class.unsqueeze(0))
					loss = loss_indiv.sum()
				
				grad += torch.autograd.grad(loss, [x_adv])[0].detach()

			grad /= float(self.eot_iter)
			if self.verbose:
				print("iteration: {} - Best loss: {:.6f}".format(i, loss_best.sum()))

			# check step size
			with torch.no_grad():
				y1 = loss_indiv.detach().clone()
				loss_steps[i] = y1.cpu() + 0
				ind = (y1 > loss_best).nonzero().squeeze()
				x_best[ind] = x_adv[ind].clone()
				grad_best[ind] = grad[ind].clone()
				loss_best[ind] = y1[ind] + 0
				loss_best_steps[i + 1] = loss_best + 0

				counter3 += 1

				if counter3 == k:
					fl_oscillation = self.check_oscillation(
						loss_steps.detach().cpu().numpy(),
						i,
						k,
						loss_best.detach().cpu().numpy(),
						k3=self.thr_decr,
					)
					fl_reduce_no_impr = (~reduced_last_check) * (
						loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
					)
					fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
					reduced_last_check = np.copy(fl_oscillation)
					loss_best_last_check = loss_best.clone()

					if np.sum(fl_oscillation) > 0:
						step_size[u[fl_oscillation]] /= 2.0

						fl_oscillation = np.where(fl_oscillation)

						x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
						grad[fl_oscillation] = grad_best[fl_oscillation].clone()

					counter3 = 0
					k = np.maximum(k - self.size_decr, self.steps_min)

		return best_state, best_q
