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
from models import MLP, NatureCNN
from camp import CAMP_loss

def none_or_int(x):
    if x.lower() == 'none':
        return None
    try:
        return int(x)
    except:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {x}")  


parser = argparse.ArgumentParser(description='Train DQN')
# Common arguments
parser.add_argument('--exp_name', type=str, default="train_dqn",
                    help='the name of this experiment')
parser.add_argument('--env_id', type=str, default="cartpole_simple",
                    choices=["cartpole_simple", "cartpole_multiframe", "pong1r", "freeway", "bankheist", "highway"],
                    help='the id of the gym environment')
parser.add_argument('--config_path', type=str, default='config')
parser.add_argument('--load_checkpoint', type=str, default='',
                    help='Where checkpoint file should be loaded from')
parser.add_argument('--seed', type=none_or_int, default=0,
                    help='fix seed for reproducibility.')
parser.add_argument('--torch_deterministic', action='store_true', default=False,
                    help='fix torch for reproducibility.')
parser.add_argument('--distill', action='store_true', default=False,
                    help='Distill policy from a trained one.')
parser.add_argument('--distill_path', type=str, default='',
                    help='Path of the model to be distilled.')
# # Algorithm specific arguments
parser.add_argument('--train_mode', default='camp',
                    choices=["camp", "baseline"],
                    help='which training method to use?')
parser.add_argument('--env_sigma', type=float, default=0.0,
                     help='noise scale for observation wrapper')
parser.add_argument('--num_models', type=int, default=1,
                     help='Number of models in the ensemble.')
parser.add_argument('--lamda', type=float, default=1.0,
                     help='CAMP coefficient')
parser.add_argument("--max_grad_norm", type=float, default=10, 
                    help="Gradient norm for clipping.")
args = parser.parse_args()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def eval(eval_env, models, num_episodes=10, max_frames_per_episode=30000):
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
        with torch.no_grad():
            q_table = torch.stack([model(state) for model in models]).sum(0)
            action = q_table.max(1).indices.view(1,1)
        next_obs, reward, terminated, truncated, _ = eval_env.step(action.item())
        done = terminated or truncated

        obs = next_obs
        episode_reward += reward
        if this_episode_frame == max_frames_per_episode:
            done = True

        if done:
            seed = random.randint(0, sys.maxsize)
            obs, _ = eval_env.reset()
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


def sample_action(env, state, policys, device):
    global eps_threshold
    global start_e
    sample = random.random()
    eps_threshold = linear_schedule(start_e, 
                                    config.eps_end, 
                                    config.exploration_fraction * config.total_timesteps, 
                                    global_step)
    # sample by policy with probability $1 - eps_threshold$
    if sample > eps_threshold and global_step > config.learning_starts:
        #print(f"Sampling by policy at global step {global_step}. Current eps_threshold: {eps_threshold}, RGN: {sample}")
        with torch.no_grad():
            q_table = torch.stack([policy(state) for policy in policys]).sum(0)
            return q_table.max(1).indices.view(1,1)
    # Exploration with probability $eps_threshold$
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def transition_step(env, buffer, state, policy_nets, device):
    action = sample_action(env, state, policy_nets, device)
    # Transit to next time step
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated
    if done:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    # Store the transition in memory
    buffer.push(state, action, next_state, reward)
    return next_state, reward, done


def camp_nll_train_step(buffer, tran, policy_nets, teacher_nets, opts, device, gradient_steps=1, N=4, train_sigma=0.4):
    [pnet.train() for pnet in policy_nets]
    [tnet.eval() for tnet in teacher_nets]
    #Start training when buffer has >=one batch of data
    if len(buffer) < config.batch_size:
        return
    # only support one model
    assert len(policy_nets) == 1

    for _ in range(gradient_steps):
        # sample from buffer
        transitions = buffer.sample(config.batch_size)
        batch = tran(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        
        policy_q_table = policy_nets[0](state_batch)
        with torch.no_grad():
            # compute teacher Q table
            teacher_q_table = teacher_nets[0](state_batch)
            teacher_actions = teacher_q_table.max(1).indices
        
        # compute logits gap
        with torch.no_grad():
            top2 = torch.topk(policy_q_table, 2)
            top2_values = top2.values
            logits_gaps = top2_values[:, 0] - top2_values[:, 1]

        criterion = nn.CrossEntropyLoss() 
        loss = criterion(policy_q_table, teacher_actions)

        eta = policy_q_table.max() - policy_q_table.min()
        reg = CAMP_loss(eta)
        reg_loss = reg.forward_v1(policy_q_table, teacher_q_table)
        total_loss = loss + args.lamda * reg_loss

        if global_step % 100 == 0:
            current_step = global_step + repeat_num * config.total_timesteps
            writer.add_scalar(f"Losses/Total_loss", total_loss, current_step)
            writer.add_scalar(f"Losses/TD_loss", loss, current_step)
            writer.add_scalar(f"Losses/CAMP_loss", reg_loss, current_step)
            writer.add_scalar(f"Losses/Eta Value", eta, current_step)
            writer.add_scalar(f"Logits_Gaps/max_logits_gap", logits_gaps.max(), current_step)
            writer.add_scalar(f"Logits_Gaps/min_logits_gap", logits_gaps.min(), current_step)
        
        opts[0].zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(policy_nets[0].parameters()), args.max_grad_norm)
        opts[0].step()


def baseline_train_step(buffer, tran, policy_nets, target_nets, opts, device, gradient_steps=1):
    [pnet.train() for pnet in policy_nets]
    [tnet.eval() for tnet in target_nets]
    #Start training when buffer has >=one batch of data
    if len(buffer) < config.batch_size:
        return
    # only support one model
    assert len(policy_nets) == 1

    for _ in range(gradient_steps):
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

        policy_q_table = policy_nets[0](state_batch)
        q_values = policy_q_table.gather(1, action_batch)
        next_state_q_table =  torch.zeros_like(policy_q_table, device=device) #(batch_size, action_size)
        next_state_values = torch.zeros(config.batch_size, device=device)     
        with torch.no_grad():
            non_final_logits = target_nets[0](non_final_next_states)
            next_state_q_table[non_final_mask] = non_final_logits
            expected_q_table = (next_state_q_table * config.gamma) + reward_batch.unsqueeze(1)
            expected_q_values = expected_q_table.max(1).values.unsqueeze(1)
        
        criterion = nn.SmoothL1Loss() # Huber loss
        
        # compute logits gap
        with torch.no_grad():
            top2 = torch.topk(policy_q_table, 2)
            top2_values = top2.values
            logits_gaps = top2_values[:, 0] - top2_values[:, 1]

        loss = criterion(q_values, expected_q_values)
        if global_step % 100 == 0:
            current_step = global_step + repeat_num * config.total_timesteps
            if args.train_mode == "camp":
                writer.add_scalar(f"Losses/RefNet_TD_Loss", loss, current_step)
            else:
                writer.add_scalar(f"Losses/TD_Loss", loss, current_step)
            writer.add_scalar(f"Logits_Gaps/max_logits_gap", logits_gaps.max(), current_step)
            writer.add_scalar(f"Logits_Gaps/min_logits_gap", logits_gaps.min(), current_step)
        
        opts[0].zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(policy_nets[0].parameters()), args.max_grad_norm)
        opts[0].step()


def main(env, ev_env, buffer, ref_buffer, tran,
         policy_nets, target_nets, ref_nets, ref_target_nets, opts, ref_opts, device, seed):
    #global config
    global eps_threshold
    global global_step
    global best_val_reward
    logger.info(f"Save model every {config.validation_interval} steps")
    
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    logger.info(f"Random seed in this run: {seed}")
    
    # Initialize the environment and get its state
    state, _ = env.reset(seed=seed)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for global_step in tqdm(range(start_step, config.total_timesteps)):
        # Primary Policy: play and memorize.
        next_state, reward, done = transition_step(env, buffer, state, policy_nets, device)
        state = next_state
        if done:
            # Reset state and start new episode
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        elif args.train_mode == "camp" and args.distill == False:
            # Reference Policy: play and memorize.
            next_state, reward, done = transition_step(env, ref_buffer, state, ref_nets, device)
            state = next_state
            if done:
                # Reset state and start new episode
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Train step
        if global_step > config.learning_starts and global_step % config.train_frequency == 0:
            if args.train_mode == "camp":
                if args.distill == False:
                    baseline_train_step(ref_buffer, tran, ref_nets, ref_target_nets, ref_opts, device, 
                                        gradient_steps=config.gradient_steps)
                
                camp_nll_train_step(buffer, tran, policy_nets, ref_nets,
                                    opts, device, 
                                    gradient_steps=config.gradient_steps)

            elif args.train_mode == "baseline":
                baseline_train_step(buffer, tran, policy_nets, target_nets, opts, device, 
                                gradient_steps=config.gradient_steps)
                
            # update target network
            if global_step % config.target_network_frequency == 0:
                target_net_state_dict = target_nets[0].state_dict()
                policy_net_state_dict = policy_nets[0].state_dict()
                # Soft update of the target network's weights
                # # θ′ ← τ θ + (1 −τ )θ′
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*config.tau + target_net_state_dict[key]*(1-config.tau)
                target_nets[0].load_state_dict(target_net_state_dict)
                
                if args.train_mode == "camp" and args.distill == False:
                    # Update the target network of the reference policy
                    ref_target_net_state_dict = ref_target_nets[0].state_dict()
                    ref_net_state_dict = ref_nets[0].state_dict()
                    # Soft update of the target network's weights
                    # # θ′ ← τ θ + (1 −τ )θ′
                    for key in ref_net_state_dict:
                        ref_target_net_state_dict[key] = ref_net_state_dict[key]*config.tau + ref_target_net_state_dict[key]*(1-config.tau)
                    ref_target_nets[0].load_state_dict(ref_target_net_state_dict)


        # Validation step
        if global_step > config.learning_starts and global_step % config.validation_interval == 0:
            current_step = global_step + repeat_num * config.total_timesteps
            test_reward=eval(ev_env, 
                             policy_nets, 
                             num_episodes=config.validation_episodes,
                             max_frames_per_episode=config.max_frames_per_val_episode)
            logger.info(f'Mean test reward: {test_reward}, at step: {global_step} or run: {repeat_num}.')
            writer.add_scalar("charts/Eval_episode_reward", test_reward, current_step)
            writer.add_scalar("charts/epsilon", eps_threshold, current_step)

            if args.train_mode == "camp":
                teacher_reward=eval(ev_env, 
                                    ref_nets, 
                                    num_episodes=config.validation_episodes,
                                    max_frames_per_episode=config.max_frames_per_val_episode)
                logger.info(f'Teacher mean test reward: {teacher_reward}, at step: {global_step}')
                writer.add_scalar("charts/Eval_episode_reward (Teacher net)", teacher_reward, current_step)
            
            if test_reward >= best_val_reward:
                logger.info(f'New best reward {test_reward} achieved at step {global_step} of run {repeat_num}, update checkpoint!')
                best_val_reward = test_reward
                torch.save({
                    'frames': global_step+1,
                    'model_state_dict': [p_net.state_dict() for p_net in policy_nets],
                    'optimizer_state_dict': [opt.state_dict() for opt in optimizers],
                    'epsilon': eps_threshold,
                }, f"./{checkpoint_path_file}" + "_restore.pth")
                logger.info(f"Model saved at: {checkpoint_path_file}.pth")
            
            if "cartpole" in args.env_id and best_val_reward == 200:
                break
    env.close()


if __name__ == "__main__":
    # Define log paths
    exp_path = args.exp_name
    exp_path = os.path.join('exp', exp_path)
    exp_path = os.path.join(exp_path, args.env_id)
    if 'baseline' in args.exp_name:
        exp_path = os.path.join(exp_path, f"wrapper_sigma={args.env_sigma}-seed={args.seed}")
    elif 'camp' in args.exp_name:
        exp_path = os.path.join(exp_path, f"wrapper_sigma={args.env_sigma}-lamda={args.lamda}-seed={args.seed}")
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
    
    # Define transition tuple
    Transition = namedtuple("Transition", 
                            ("state", "action", "next_state", "reward")
                            )

    # Repeat loop
    best_val_reward = -float('inf')
    for i in range(config.num_repeat_exp):
        # Setup environments
        env = setup_env(args.env_id, args.env_sigma, train_video_path, 
                        save_freq=config.total_timesteps//10, epsodic_life=False)
        eval_env = setup_env(args.env_id, args.env_sigma, eval_video_path, 
                            save_freq=config.total_timesteps//100, epsodic_life=False)
        
        logger.info(f"Enviroment: {args.env_id}")
        logger.info("="*40)
        
        # Define model
        state, _ = env.reset()
        print("Check state:", state, state.shape)
        act_dim = env.action_space.n
        if "cartpole" in args.env_id or "highway" in args.env_id:
            stat_dim = len(state)
            dqn_ensemble = [MLP(stat_dim, act_dim).to(device) for _ in range(args.num_models)]
            target_ensemble = [MLP(stat_dim, act_dim).to(device) for _ in range(args.num_models)]
            ref_ensemble = [MLP(stat_dim, act_dim).to(device) for _ in range(args.num_models)]
            ref_target_ensemble = [MLP(stat_dim, act_dim).to(device) for _ in range(args.num_models)]
        else:
            stat_dim = np.array(state).shape
            dqn_ensemble = [NatureCNN(stat_dim[0], act_dim).to(device) for _ in range(args.num_models)]
            target_ensemble = [NatureCNN(stat_dim[0], act_dim).to(device) for _ in range(args.num_models)]
            ref_ensemble = [NatureCNN(stat_dim[0], act_dim).to(device) for _ in range(args.num_models)]
            ref_target_ensemble = [NatureCNN(stat_dim[0], act_dim).to(device) for _ in range(args.num_models)]
        optimizers = [optim.Adam(dqn.parameters(), lr=config.lr) for dqn in dqn_ensemble]
        ref_optimizers = [optim.Adam(refnet.parameters(), lr=config.lr) for refnet in ref_ensemble]
        logger.info(f"State dimension: {stat_dim}, Action dimension: {act_dim}")
        logger.info("="*40)

        # Define replay buffer
        memory = ReplayBuffer(config.buffer_size, Transition)
        ref_memory = ReplayBuffer(config.buffer_size, Transition)

        start_step=0
        start_e = config.eps_start
        
        # training from checkpoint
        if args.load_checkpoint:
            logger.info(f"Loading a policy - {args.load_checkpoint} ")
            checkpoint = torch.load(args.load_checkpoint)
            [dqn.load_state_dict(
                checkpoint['model_state_dict'][i]) for i, dqn in enumerate(dqn_ensemble)]
            [optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'][i]) for i, optimizer in enumerate(optimizers)]
            #start_step=checkpoint['frames']
            #start_e=checkpoint['epsilon']
            assert start_e >= config.eps_end
            logger.info(f'Primary policy training from global step:{start_step}, epsilon:{start_e}')
        [target_net.load_state_dict(dqn.state_dict()) for dqn, target_net in zip(dqn_ensemble, target_ensemble)]

        # load pretrained policy for distillation
        if args.distill:
            logger.info(f"Loading a policy to distill - {args.distill_path} ")
            dist_ckpt = torch.load(args.distill_path)
            [refnet.load_state_dict(dist_ckpt['model_state_dict'][i]) for i, refnet in enumerate(ref_ensemble)]
            [refnet.eval() for refnet in ref_ensemble]
            logger.info("Teacher net loaded.")
            debug_reward=eval(eval_env, 
                            ref_ensemble, 
                            num_episodes=config.validation_episodes,
                            max_frames_per_episode=config.max_frames_per_val_episode)
            
            logger.info(f"Teacher net evaluation reward averaged from {config.validation_episodes} runs: {debug_reward}")
        [ref_target_net.load_state_dict(refnet.state_dict()) for refnet, ref_target_net in zip(ref_ensemble, ref_target_ensemble)]
        
        repeat_num = i
        if not args.seed == 99:
            new_seed = args.seed + i
        else:
            new_seed = None
        main(env, eval_env, 
             memory, ref_memory,
             Transition,
             dqn_ensemble, target_ensemble, 
             ref_ensemble, ref_target_ensemble,
             optimizers, ref_optimizers,
             device, new_seed)
    writer.close()