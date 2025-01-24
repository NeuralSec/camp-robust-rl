import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def randomize_states(state, sigma, num_copy):
    state_set = state.unsqueeze(0).repeat(num_copy)
    return state_set + sigma * torch.randn_like(state_set)


class CAMP_loss(nn.Module):
    def __init__(self, eta):
        super(CAMP_loss, self).__init__()
        self.eta = eta
    
    def get_networks(self, policy, target):
        self.policy_net = policy
        self.target_net = target

    def forward_v1(self, q_table, expected_q_table):
        top2 = torch.topk(q_table, 2)
        top2_values = top2.values
        top2_actions = top2.indices
        target_top1_actions = expected_q_table.max(1).indices
        indx_selected = (top2_actions[:, 0] == target_top1_actions).bool()
        logits_gap = top2_values[indx_selected, 1] - top2_values[indx_selected, 0] # negative values
        indx = ~torch.isnan(logits_gap) & ~torch.isinf(logits_gap) & (torch.abs(logits_gap) <= self.eta)
        camp_loss = logits_gap[indx] + self.eta
        if len(camp_loss) != 0:
            return camp_loss.mean()
        else:
            return camp_loss.sum()
        