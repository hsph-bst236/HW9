"""
---
title: Mario wrapper with multi-processing
summary: This implements the Mario game environment with multi-processing.
---

# Mario wrapper with multi-processing
"""
import multiprocessing
import multiprocessing.connection

import gym
import numpy as np

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils.gym_utils import SMBRamWrapper


class Mario:
    """
    ## Game environment

    This is a wrapper for Super Mario Bros game environment.
    We do a few things here:

    1. Apply the same action for n_skip frames
    2. Stack n_stack frames of the last n_stack actions
    3. Add episode information (total reward for the entire episode) for monitoring
    4. Crop the observation space to focus on relevant game area

    #### Observation format
    Observation is tensor of size (n_stack, height, width). It is n_stack frames
    stacked on first axis. i.e, each channel is a frame.
    """

    def __init__(self, 
                 name='SuperMarioBros-1-1-v0', 
                 seed=47,
                 n_stack=4,
                 n_skip=4,
                 crop_size=[0, 16, 0, 13]):
        """
        Initialize Mario environment
        Args:
            name: name of the gym environment
            seed: random seed
            n_stack: number of frames to stack
            n_skip: number of frames to skip
            crop_size: [x_min, x_max, y_min, y_max] for cropping the observation
        """
        # create environment
        self.env = gym_super_mario_bros.make(name)
        self.env.seed(seed)

        # wrap environment for frame stacking and cropping
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        #self.env = JoypadSpace(self.env,  [["right"], ["right", "A"]])
        self.env = SMBRamWrapper(self.env, crop_size, n_stack=n_stack, n_skip=n_skip)

        # tensor for a stack of n frames
        self.obs_shape = (n_stack, crop_size[3], crop_size[1])
        self.obs = np.zeros(self.obs_shape)

        # keep track of the episode rewards
        self.rewards = []
        
        # Store parameters
        self.n_stack = n_stack
        self.n_skip = n_skip
        self.crop_size = crop_size

    def step(self, action):
        """
        ### Step
        Executes `action` for n_skip time steps and
         returns a tuple of (observation, reward, done, episode_info).

        * `observation`: stacked n frames 
        * `reward`: total reward while the action was executed
        * `done`: whether the episode finished
        * `episode_info`: episode information if completed
        """
        obs, reward, done, info = self.env.step(action)
        obs = np.transpose(obs, (2, 0, 1))  # Reshape to (n_stack, height, width)

        # maintain rewards for each step
        self.rewards.append(reward)

        if done:
            # if finished, set episode information and reset
            episode_info = {
                "reward": sum(self.rewards), 
                "length": len(self.rewards)
            }
            self.reset()
        else:
            episode_info = None

        return obs, reward, done, episode_info

    def reset(self):
        """
        ### Reset environment
        Clean up episode info and frame stack
        """
        # reset OpenAI Gym environment
        obs = self.env.reset()
        obs = np.transpose(obs, (2, 0, 1))  # Reshape to (n_stack, height, width)
        # reset caches
        self.rewards = []

        return obs

    @property
    def action_space(self):
        """Get the action space of the environment"""
        return self.env.action_space

    @property
    def observation_space(self):
        """Get the observation space of the environment"""
        return self.env.observation_space


def worker_process(remote: multiprocessing.connection.Connection, seed: int):
    """
    ## Worker Process

    Each worker process runs this method. It creates a game instance and
    executes commands received through the connection.
    
    Args:
        remote: Connection to communicate with the parent process
        seed: Random seed for the game environment
    """
    # create game
    game = Mario(seed=seed)

    # wait for instructions from the connection and execute them
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class Worker:
    """
    Creates a new worker and runs it in a separate process.
    
    This class handles the creation of a game environment in a separate process
    and provides a way to communicate with it through a pipe connection.
    """

    def __init__(self, seed: int):
        """
        Initialize a new worker process
        
        Args:
            seed: Random seed for the game environment
        """
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()


