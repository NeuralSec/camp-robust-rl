import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import os 
import sys
import numpy as np
import random
from collections import namedtuple
from tqdm import tqdm
import mlconfig
from util import setup_env, ReplayBuffer, setup_logger, build_dirs
from models import NoisyNet_MLP, NoisyNet


parser = argparse.ArgumentParser(description='Train DQN with NoisyNet')
# Common arguments
parser.add_argument('--exp_name', type=str, default="train_dqn_noisynet",
                    help='the name of this experiment')
parser.add_argument('--env_id', type=str, default="cartpole_simple", 
                    choices=["cartpole_simple", "cartpole_multiframe", "pong1r", "freeway", "bankheist", "highway"],
                    help='the id of the gym environment')
parser.add_argument('--config_path', type=str, default='config')
parser.add_argument('--load_checkpoint', type=str, default='',
                    help='Where checkpoint file should be loaded from')
parser.add_argument('--seed', type=int, default=0,
                    help='fix seed for reproducibility.')
parser.add_argument('--torch_deterministic', action='store_true', default=False,
                    help='fix torch for reproducibility.')
parser.add_argument('--restrict_actions', type=int, default=0,
                    help='restrict actions or not')
# # Algorithm specific arguments
parser.add_argument('--train_mode', default='baseline', 
                    choices=["baseline"],
                    help='which training method to use?')
parser.add_argument('--env_sigma', type=float, default=0.0,
                     help='noise scale for observation wrapper')     
parser.add_argument("--max_grad_norm", type=float, default=10, 
                    help="Gradient norm for clipping.")
args = parser.parse_args()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def eval(eval_env, model, num_episodes=10, max_frames_per_episode=30000):
    logger.info("="*20 + 'Start Eval' + "="*20)
    
    all_rewards = []
    episode_reward = 0
    
    seed = random.randint(0, sys.maxsize)
    obs, _ = eval_env.reset(seed=seed)

    episode_idx = 1
    this_episode_frame = 1
    for frame_idx in range(1, num_episodes * max_frames_per_episode + 1):
        obs = np.array(obs)
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        if "freeway" in args.env_id or "pong" in args.env_id:
            state /= 255
        with torch.no_grad():
            q_table = model(state)
            action = q_table.max(1).indices.view(1,1)
        next_obs, reward, terminated, truncated, _ = eval_env.step(action.item())
        done = terminated or truncated

        obs = next_obs
        episode_reward += reward
        if this_episode_frame == max_frames_per_episode:
            done = True

        if done:
            seed = random.randint(0, sys.maxsize)
            obs, _ = eval_env.reset(seed=seed)
            all_rewards.append(episode_reward)
            logger.info('episode {}/{} reward: {:6g}'.format(episode_idx, num_episodes, all_rewards[-1]))
            episode_reward = 0
            this_episode_frame = 1
            episode_idx += 1
            if episode_idx > num_episodes:
                break
        else:
            this_episode_frame += 1
    return np.mean(all_rewards)

def sample_action(env, state, policy, device):
    global global_step
    global eps_threshold
    sample = random.random()
    eps_threshold = linear_schedule(config.eps_start, 
                                    config.eps_end, 
                                    config.exploration_fraction * config.total_timesteps, 
                                    global_step)
    # sample by policy
    if sample > eps_threshold and global_step > config.learning_starts:
        with torch.no_grad():
            policy.sample()
            q_table = policy(state)
            return q_table.max(1).indices.view(1,1)
    # Exploration with probability $eps_threshold$
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def baseline_train_step(buffer, tran, p_net, t_net, opt, device, gradient_steps=1):
    global global_step
    if len(buffer) < config.batch_size:
        return
    
    for _ in range(gradient_steps):
        p_net.sample()
        t_net.sample()
        
        # sample from buffer
        transitions = buffer.sample(config.batch_size)
        batch = tran(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        policy_q_table = p_net(state_batch)
        q_values = policy_q_table.gather(1, action_batch)
        next_state_q_table =  torch.zeros_like(policy_q_table, device=device) #(batch_size, action_size)   
        with torch.no_grad():
            non_final_logits = t_net(non_final_next_states)
            next_state_q_table[non_final_mask] = non_final_logits
            expected_q_table = (next_state_q_table * config.gamma) + reward_batch.unsqueeze(1)
            expected_q_values = expected_q_table.max(1).values.unsqueeze(1)

        
        criterion = nn.SmoothL1Loss() # Huber loss
        loss = criterion(q_values, expected_q_values)

        with torch.no_grad():
            top2 = torch.topk(policy_q_table, 2)
            top2_values = top2.values
            logits_gaps = top2_values[:, 0] - top2_values[:, 1]

        if global_step % 100 == 0:
            writer.add_scalar(f"Losses/TD_loss", loss, global_step)
            writer.add_scalar(f"Losses/max_logits_gap", logits_gaps.max(), global_step)
            writer.add_scalar(f"Losses/min_logits_gap", logits_gaps.min(), global_step)
        
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(p_net.parameters()), args.max_grad_norm)
        opt.step()


def main(env, ev_env, buffer, tran, policy_net, target_net, opt, device, seed):
    #global config
    global eps_threshold
    global global_step
    global best_val_reward
    logger.info(f"Save model every {config.validation_interval} steps")
    
    episode_reward = 0
    episode_id = 0
    
    # Initialize the environment and get its state
    state, _ = env.reset(seed=seed)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for global_step in tqdm(range(start_step, config.total_timesteps)):
        action = sample_action(env, state, policy_net, device)
        # Transit to next time step
        observation, reward, terminated, truncated, _ = env.step(action.item())
        episode_reward += reward
        reward = torch.tensor([reward], device=device)
        # logger.info(f"Current reward: {reward.item()}")
        done = terminated or truncated

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Store the transition in memory
        buffer.push(state, action, next_state, reward)
        # Move to the next state, state here is already normalized
        state = next_state

        # Train step
        if global_step > config.learning_starts and global_step % config.train_frequency == 0:
            baseline_train_step(buffer, tran, policy_net, target_net, opt, device, 
                                gradient_steps=config.gradient_steps)
            
            # update target network
            if global_step % config.target_network_frequency == 0:
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                # Soft update of the target network's weights
                # # θ′ ← τ θ + (1 −τ )θ′
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*config.tau + target_net_state_dict[key]*(1-config.tau)
                target_net.load_state_dict(target_net_state_dict)
                
        if done:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            if global_step > config.learning_starts:
                writer.add_scalar("charts/Train_episode_reward", episode_reward, global_step)
            episode_id += 1
            episode_reward = 0

        if global_step > config.learning_starts and global_step % config.validation_interval == 0:
            test_reward=eval(ev_env, 
                             policy_net, 
                             num_episodes=config.validation_episodes,
                             max_frames_per_episode=config.max_frames_per_val_episode)
            logger.info(f'Mean test reward: {test_reward}, at step: {global_step}')
            writer.add_scalar("charts/Eval_episode_reward", test_reward, global_step)
            writer.add_scalar("charts/epsilon", eps_threshold, global_step)
            
            if test_reward >= best_val_reward:
                logger.info(f'New best reward {test_reward} achieved at step {global_step}, run {config.num_repeat_exp}, update checkpoint!')
                best_val_reward = test_reward
                torch.save({
                    'frames': global_step+1,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'epsilon': eps_threshold,
                }, f"./{checkpoint_path_file}" + "_restore.pth")
                logger.info(f"Model saved at: {checkpoint_path_file}.pth")
            
            if "cartpole" in args.env_id and best_val_reward == 200:
                break
    env.close()
    writer.close()


if __name__ == "__main__":
    # Define log paths
    exp_path = args.exp_name
    exp_path = os.path.join('exp', exp_path)
    exp_path = os.path.join(exp_path, args.env_id)
    exp_path = os.path.join(exp_path, f"wrapper_sigma={args.env_sigma}-seed={args.seed}")
    checkpoint_path = os.path.join(exp_path, 'checkpoints')
    checkpoint_path_file = os.path.join(checkpoint_path, args.env_id)
    log_path = os.path.join(exp_path, "log")
    train_video_path = os.path.join(exp_path, "train_video")
    eval_video_path = os.path.join(exp_path, "eval_video")
    log_path_file = os.path.join(log_path, "log")
    build_dirs(exp_path)
    build_dirs(checkpoint_path)
    build_dirs(log_path)
    build_dirs(train_video_path)
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
    env = setup_env(args.env_id, args.env_sigma, train_video_path, 
                    save_freq=config.total_timesteps//10)
    eval_env = setup_env(args.env_id, args.env_sigma, eval_video_path, 
                         save_freq=config.total_timesteps//100)
    # Define transition tuple
    Transition = namedtuple("Transition", 
                            ("state", "action", "next_state", "reward")
                            )
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    logger.info(f"Enviroment: {args.env_id}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("="*40)
    
    # Define model
    state, _ = env.reset(seed=args.seed)
    act_dim = env.action_space.n
    if "cartpole" in args.env_id or "highway" in args.env_id:
        stat_dim = len(state)
        dqn = NoisyNet_MLP(env).to(device)
        target = NoisyNet_MLP(env).to(device)
    else:
        stat_dim = np.array(state).shape
        dqn = NoisyNet(env).to(device)
        target = NoisyNet(env).to(device)
    optimizer = optim.Adam(dqn.parameters(), lr=config.lr)
    logger.info(f"State dimension: {stat_dim}, Action dimension: {act_dim}")
    logger.info("="*40)
    
    start_step=0
    # training from checkpoint
    if args.load_checkpoint:
        logger.log(f"Loading a policy - {args.load_checkpoint} ")
        checkpoint = torch.load(args.load_checkpoint)
        dqn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step=checkpoint['frames']
        start_e=checkpoint['epsilon']
        logger.log(f'training from frames:{start_step},epsilon:{start_e}')

        if args.start_e>args.end_e:
            temp_fraction=args.exploration_fraction
            args.exploration_fraction=(start_e-args.end_e)*temp_fraction/(args.start_e-args.end_e)
            args.start_e=start_e

    target.load_state_dict(dqn.state_dict())
    
    
    # Define replay buffer
    memory = ReplayBuffer(config.buffer_size, Transition)

    # train
    best_val_reward = -float('inf')
    for i in range(config.num_repeat_exp):
        new_seed = args.seed + i
        main(env, eval_env, memory, Transition, 
             dqn, target, 
             optimizer,
             device, 
             new_seed)