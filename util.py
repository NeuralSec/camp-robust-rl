import logging
import os
import numpy as np
import torch
import random
from collections import deque
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, FrameStack, FlattenObservation, AtariPreprocessing, RecordVideo
from wrappers import NoisyObsWrapper, FinishEarlyWrapper
import highway_env


def setup_env(env_id, env_sigma, video_path, save_freq=200, epsodic_life=False):
    # Initialize env
    if env_id == "freeway":
        env = gym.make("FreewayNoFrameskip-v0", render_mode="rgb_array", difficulty=1)
        env = AtariPreprocessing(env, scale_obs=True, terminal_on_life_loss=epsodic_life)
        env = RecordVideo(
            TimeLimit(FrameStack(NoisyObsWrapper(env, env_sigma), 4) ,250), 
            video_path, 
            step_trigger= lambda x: x % save_freq == 0
            )
    elif env_id == "cartpole_simple":
        env = RecordVideo(
            NoisyObsWrapper(gym.make("CartPole-v0", render_mode="rgb_array"), env_sigma), 
            video_path, 
            step_trigger= lambda x: x % save_freq == 0
            )
    elif env_id == "cartpole_multiframe":
        env = RecordVideo(
            FlattenObservation(FrameStack(NoisyObsWrapper(gym.make("CartPole-v0", render_mode="rgb_array"), env_sigma), 5) ), 
            video_path, 
            step_trigger= lambda x: x % save_freq == 0
            )
    elif env_id == "pong1r":
        env = gym.make("PongNoFrameskip-v0", render_mode="rgb_array")
        env = AtariPreprocessing(env, scale_obs=True, terminal_on_life_loss=epsodic_life)
        env = RecordVideo(
            FrameStack(NoisyObsWrapper(FinishEarlyWrapper(env), env_sigma), 4), 
            video_path, 
            step_trigger= lambda x: x % save_freq == 0
            )
    elif env_id == "bankheist":
        env = gym.make("BankHeistNoFrameskip-v4", render_mode="rgb_array", difficulty=1)
        env = AtariPreprocessing(env, scale_obs=True, terminal_on_life_loss=epsodic_life)
        env = RecordVideo(
            TimeLimit(FrameStack(NoisyObsWrapper(env, env_sigma), 4) ,250), 
            video_path, 
            step_trigger= lambda x: x % save_freq == 0
            )
    elif env_id == "highway":
        env = gym.make('highway-fast-v0', config={"lanes_count": 3})
        env = NoisyObsWrapper(env, env_sigma)
        env = FlattenObservation(env)
    else:
        raise NotImplementedError
    return env


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


class ReplayBuffer():
    def __init__(self, buffer_limit, Transition):
        self.Transition = Transition
        self.buffer = deque([], maxlen=buffer_limit)

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
