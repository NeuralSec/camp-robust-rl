import torch
import argparse
import numpy as np
import mlconfig
import os
import sys
import random
from torch.utils.tensorboard import SummaryWriter
from util import setup_env, setup_logger, build_dirs
from models import NoisyNet_MLP, NoisyNet


parser = argparse.ArgumentParser(description='DQN with NoisyNet Test')
parser.add_argument('--exp_name', type=str, default="test_dqn",
                    help='the name of this experiment')
parser.add_argument('--env_id', type=str, default="cartpole_simple", 
                    choices=["cartpole_simple", "cartpole_multiframe", "pong1r", "freeway", "bankheist", "highway"],
                    help='the id of the gym environment')
parser.add_argument('--config_path', type=str, default='config')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Where checkpoint file should be loaded from')
parser.add_argument('--seed', type=int, default=0,
					help='random seed (default: 0)')
parser.add_argument('--torch_deterministic', action='store_true', default=False,
                    help='fix torch for reproducibility.')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--env_sigma', type=float, default=0.0,
                     help='noise scale for observation wrapper')
parser.add_argument('--num_evals', type=int, default=10,
					help='Number of evaluations')
parser.add_argument('--store_all_rewards', action='store_true',
					help='store all rewards (vs just sum)')
parser.add_argument('--no_save', action='store_true',
					help='Not saving cumulative rewards as pth file')
args = parser.parse_args()



if __name__ == '__main__':
	# Define log paths
    exp_path = args.exp_name
    exp_path = os.path.join('eval_exp', exp_path)
    exp_path = os.path.join(exp_path, args.env_id)
    exp_path = os.path.join(exp_path, f"wrapper_sigma={args.env_sigma}-eval")
    log_path = os.path.join(exp_path, "log")
    eval_result_file_path = os.path.join(exp_path, 'eval_results')
    eval_video_path = os.path.join(exp_path, "eval_video")
    log_path_file = os.path.join(log_path, "log")
    build_dirs(exp_path)
    build_dirs(log_path)
    build_dirs(eval_video_path)
    logger = setup_logger(name=args.env_id, log_file=log_path_file + ".log")
    writer = SummaryWriter(log_path + "/tblog")

    # Device Options
    logger.info("="*50)
    logger.info("PyTorch Version: %s" % (torch.__version__))
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    else:
        device = torch.device('cpu')

    # Load and log exp configs
    config_file = os.path.join(args.config_path, args.env_id)+'.yaml'
    config = mlconfig.load(config_file)
    config.set_immutable()
    for key in config:
        logger.info("%s: %s" % (key, config[key]))
    logger.info("="*40)

    # Record hyper-parameters
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    logger.info("="*40)
    writer.add_text('Args', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    writer.add_text('Hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{config[key]}|" for key in config])))

    # Setup environments
    if "cartpole" in args.env_id or "highway" in args.env_id:
        video_freq = args.num_evals * 200
    else:
        video_freq = args.num_evals * 250
    eval_env = setup_env(args.env_id, args.env_sigma, eval_video_path, 
                         save_freq= video_freq // 10)
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    eval_env.action_space.seed(args.seed)
    eval_env.observation_space.seed(args.seed)
    logger.info(f"Enviroment: {args.env_id}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("="*40)
    
    # Define model
    state, _ = eval_env.reset(seed=args.seed)
    act_dim = eval_env.action_space.n
    if "cartpole" in args.env_id or "highway" in args.env_id:
        stat_dim = len(state)
        dqn = NoisyNet_MLP(eval_env).to(device)
    else:
        stat_dim = np.array(state).shape
        dqn = NoisyNet(eval_env).to(device)
    logger.info(f"State dimension: {stat_dim}, Action dimension: {act_dim}")
    logger.info("="*40)
    
    # load checkpoint
    # training from checkpoint
    if args.checkpoint_path:
        logger.info(f"Loading a policy - {args.checkpoint_path} ")
        checkpoint = torch.load(args.checkpoint_path)
        dqn.load_state_dict(checkpoint['model_state_dict'])
    dqn.sample() # sample weights
    dqn.eval()
    reward_accum = []
    state_hist = []

    logits_gap_recorder = dict({"min": torch.tensor([1e10]).to(device), 
                               "mean": torch.tensor([0]).to(device), 
                               "max": torch.tensor([0]).to(device)
                               })
    
    for i_episode in range(args.num_evals):
        #current_seed = args.seed + i_episode
        current_seed = random.randint(0, sys.maxsize)
        ep_reward =0
        per_step_rewards = []
        this_episode_frame = 1

        obs, _ = eval_env.reset(seed=current_seed)
        state_hist.append(obs)
        done = None
        
        while not done: 
            obs = np.array(obs)
            state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_table = dqn(state)

                if "cartpole" in args.env_id:
                    top2 = torch.topk(q_table, 2)
                    top2_values = top2.values
                    logits_gaps = top2_values[:, 0] - top2_values[:, 1]
                    #print("logits_gaps shape:", logits_gaps.shape)
                    logits_gap_recorder["min"] = torch.minimum(logits_gap_recorder["min"], logits_gaps.min())
                    logits_gap_recorder["max"] = torch.maximum(logits_gap_recorder["max"], logits_gaps.max())
                    logits_gap_recorder["mean"] = (logits_gap_recorder["mean"] + logits_gaps) / 2

                action = q_table.max(1).indices.view(1,1)
            next_obs, reward, terminated, truncated, _ = eval_env.step(action.item())
            done = terminated or truncated

            obs = next_obs
            state_hist.append(obs)
            if args.render:
                eval_env.render()
            per_step_rewards.append(reward)
            ep_reward += reward

            this_episode_frame += 1
            if this_episode_frame == config.max_frames_per_val_episode:
                done = True

        if (args.store_all_rewards):
            reward_accum.append(per_step_rewards)
        else:
            reward_accum.append(ep_reward)
    
    logger.info(f"The list of cumulative rewards from {args.num_evals} eval runs: {reward_accum}.")
    if not args.store_all_rewards:
        mean_cr = torch.tensor(reward_accum).float().mean().item()
        logger.info(f"Mean cumulative reward: {mean_cr}")

    if not args.no_save:
        torch.save(reward_accum, eval_result_file_path + '.pth')
    
    if "cartpole" in args.env_id:
        for k in logits_gap_recorder.keys():
            logits_gap_recorder[k] = logits_gap_recorder[k].cpu().numpy()
        logger.info("Q value gap statistics:")
        logger.info(logits_gap_recorder)
