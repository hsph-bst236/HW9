# Define custom environment to crop obs frame
from gym import spaces
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY 
from nes_py.wrappers import JoypadSpace
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)


from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import obs_as_tensor

from .smb_utils import *

import time
import os

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation

import imageio
import torch
import numpy as np


class SMBRamWrapper(gym.ObservationWrapper):
    def __init__(self, env, crop_dim=[0, 16, 0, 13], n_stack=4, n_skip=2):
        '''
        crop_dim: [x0, x1, y0, y1]
        obs shape = (height, width, n_stack), n_stack=0 is the most recent frame
        n_skip: e.g. n_stack=4, n_skip=2, use frames [0, 2, 4, 6]
        '''
        gym.Wrapper.__init__(self, env)
        self.crop_dim = crop_dim
        self.n_stack = n_stack
        self.n_skip = n_skip
        # Modified from stable_baselines3.common.atari_wrappers.WarpFrame()
        # https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#AtariWrapper
        self.width = crop_dim[1] - crop_dim[0]
        self.height = crop_dim[3] - crop_dim[2]
        self.observation_space = spaces.Box(
            low=-1, high=2, shape=(self.height, self.width, self.n_stack), dtype=int
        )
        
        self.frame_stack = np.zeros((self.height, self.width, (self.n_stack-1)*self.n_skip+1))
        #self.INDEX_SKIP = 1
        
    def observation(self, obs):
        grid = smb_grid(self.env)
        frame = grid.rendered_screen # 2d array
        frame = self.crop_obs(frame)
        
        self.frame_stack[:,:,1:] = self.frame_stack[:,:,:-1] # shift frame_stack by 1
        self.frame_stack[:,:,0] = frame # add current frame to stack
        obs = self.frame_stack[:,:,::self.n_skip]
        return obs
    
    def reset(self):
        obs = self.env.reset()
        self.frame_stack = np.zeros((self.height, self.width, (self.n_stack-1)*self.n_skip+1))
        grid = smb_grid(self.env)
        frame = grid.rendered_screen # 2d array
        frame = self.crop_obs(frame)
        for i in range(self.frame_stack.shape[-1]):
            self.frame_stack[:,:,i] = frame
        obs = self.frame_stack[:,:,::self.n_skip]
        return obs

    def crop_obs(self, im):
        '''
        Crop observed frame image to reduce input size
        Returns cropped_frame = original_frame[y0:y1, x0:x1]
        '''
        [x0, x1, y0, y1] = self.crop_dim
        im_crop = im[y0:y1, x0:x1]
        return im_crop


# Function to save an episode as GIF
def run_episode_save_gif(env, model, gamma=0.99, filename="results/mario_final_replay.gif", sleep_time=0.02, duration = 20):
    """
    Run an episode with frame recording and then save the frames as a GIF.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    frames = []
    reset_result = env.reset()

    # Handle different Gym API versions
    if isinstance(reset_result, tuple):  # New Gym API (>=0.26)
        state = reset_result[0]  # Extract just the observation
    else:  # Old Gym API
        state = reset_result
        
    done = False
    
    while not done:
        # Get the rendered frame from the environment
        frame = env.render(mode="rgb_array")
        
        # Skip frame if it's None
        if frame is not None:
            frames.append(frame.copy())
        
        # Ensure state is a contiguous numpy array
        if isinstance(state, np.ndarray):
            state = np.ascontiguousarray(state)
        elif hasattr(state, 'numpy'):  # If it's a torch tensor
            state = np.ascontiguousarray(state.numpy())
        
        action, _ = model.predict(state, deterministic=True)
        
        if gym.__version__ >= "0.26":
            # Try new Gym API (returning 5 values)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
        else:
            # Fall back to older Gym API (returning 4 values)
            next_state, reward, done, info = env.step(action.item())

        state = next_state
        # time.sleep(sleep_time)
    
    # Only save GIF if we captured valid frames
    if frames:
        try:
            imageio.mimsave(filename, frames, duration=duration)
            print(f"Saved replay GIF to: {filename}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    else:
        print(f"No frames captured, GIF not saved.")    

def load_smb_env(name='SuperMarioBros-1-1-v0', crop_dim=[0,16,0,13], n_stack=2, n_skip=4):
    '''
    Wrapper function for loading and processing smb env
    '''
    env = gym_super_mario_bros.make(name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env_wrap = SMBRamWrapper(env, crop_dim, n_stack=n_stack, n_skip=n_skip)
    env_wrap = DummyVecEnv([lambda: env_wrap])
    
    return env_wrap

def play_episode(env, model, episodes=1, deterministic=True, render=True, return_eval=False):
    """
    Play the policy in the environment.
    
    Args:
        env: The environment to play in
        model: The model to use
        episodes: Number of episodes to play
        deterministic: Whether to use deterministic actions
        render: Whether to render the environment
        return_eval: Whether to return evaluation metrics
    
    Returns:
        Evaluation metrics if return_eval is True
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        states = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            if render:
                env.env.render()
            
            # Create a copy of the state and add batch dimension if needed
            x = torch.tensor(states, dtype=torch.float32)
            if len(x.shape) == 3:  # If no batch dimension
                x = x.unsqueeze(0)
            
            # Get action from model
            with torch.no_grad():
                try:
                    # Try using the predict method if available
                    action, _ = model.predict(x, deterministic=deterministic)
                except (AttributeError, TypeError):
                    # Otherwise use the forward method
                    pi, _ = model(x)
                    if deterministic:
                        action = pi.probs.argmax(dim=-1).cpu().numpy()[0]
                    else:
                        action = pi.sample().cpu().numpy()[0]
            
            # Step environment
            states, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    
    if return_eval:
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'avg_reward': avg_reward,
            'avg_length': avg_length
        }

class SMB():
    '''
    Wrapper function containing the processed environment and the loaded model
    '''
    def __init__(self, env, model):
        self.env = env
        self.model = model
    
    def play(self, episodes=5, deterministic=False, render=True, return_eval=False):
        for episode in range(1, episodes+1):
            states = self.env.reset()
            done = False
            score = 0
            
            if render == True:
                while not done:
                    self.env.env.render()
                    pi, _ = self.model(states)
                    if deterministic == False:
                        action = pi.sample()
                    else:
                        action = pi.argmax()

                    states, reward, done, info = self.env.step(action)
                    score += reward
                    time.sleep(0.01)
                print('Episode:{} Score:{}'.format(episode, score))
            else:
                while not done:
                    pi, _ = self.model(states)
                    if deterministic == False:
                        action = pi.sample()
                    else:
                        action = pi.argmax()

                    states, reward, done, info = self.env.step(action)
                    score += reward            
        if return_eval == True:
            return score, info
        else:
            return 
    
    def evaluate(self, episodes=20, deterministic=False):
        '''
        returns rewards, steps (both have length [episodes])
        '''
        rewards, steps = evaluate_policy(self.model, self.env, n_eval_episodes=episodes, 
                                 deterministic=deterministic, render=False, 
                                 return_episode_rewards=True)
        return rewards, steps
    
    import numpy as np


    def predict_proba(self, state):
        '''
        Predict the probability of each action given a state
        https://stackoverflow.com/questions/66428307/how-to-get-action-propability-in-stable-baselines-3/70012691#70012691?newreg=bd5479b970664069b359903e0151b4a1
        '''
        model = self.model
        obs = obs_as_tensor(state, model.policy.device)
        dis = model.policy.get_distribution(obs)
        probs = dis.distribution.probs
        probs_np = probs.detach().numpy()
        return probs_np
    
    #############
    # functions for making plots & videos
    
    def make_video_frames(self, deterministic=False):
        '''
        For each step, plot obs & rendered screen in one figure for making videoes
        '''
        state = self.env.reset()
        done = False
        score = [0]
        #self._make_combined_plot2(state, score, prob_actions)
        #self._make_combined_plot(state, score)
        
        
        while not done:
        #for i in range(1):
            prob_actions = self.predict_proba(state)
            action, _ = self.model.predict(state, deterministic=deterministic)
            state, reward, done, info = self.env.step(action)
            score += reward
            self._make_combined_plot2(state, score, prob_actions)
            #self._make_combined_plot(state, score)
 
    
    def _make_combined_plot2(self, state, score, prob_actions):
        '''
        Originally made for n_stack = 4 & n_skip = 4, SIMPLE_MOVEMENT
        '''
        # get rendered screen
        im_render = self.env.render(mode="rgb_array")
        
        n_stack = state.shape[-1]
        cmap = colors.ListedColormap(['red', 'skyblue', 'brown', 'blue'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        #obs_loc = [[0, 1], [0, 2], [1, 1], [1, 2]]
        obs_loc = [[0, 1], [1, 1], [2, 1], [3, 1]]
        obs_text = ['t (current frame)', 't-4', 't-8', 't-12']
        action_list = ['NOOP', 'right', 'right+A', 'right+B', 'right+A+B', 'A', 'left']
        
        
        ##########
        fig = plt.figure(dpi=100, figsize=(6, 6), constrained_layout=False, tight_layout=True)
        gs = fig.add_gridspec(4, 2, width_ratios=[3, 1])
        
        # individual obs frames
        for n in range(n_stack):
            ax = fig.add_subplot(gs[obs_loc[n][0], obs_loc[n][1]])
            im = ax.imshow(state[0,:,:,n], cmap=cmap, norm=norm)
            ax.set_axis_off()
            ax.text(-0.5, 14.5, obs_text[n])
        
        # prob_actions
        ax = fig.add_subplot(gs[3, 0])
        ax.bar(action_list, prob_actions[0])
        plt.xticks(rotation=45)
        ax.set_ylim(0, 1.05)

        # rendered screen
        ax = fig.add_subplot(gs[0:3, 0])
        im = ax.imshow(im_render)
        ax.set_axis_off()
        ax.text(0, -5, 'score: '+str(int(score[0])))
        
        plt.show()    
 
    
    def _make_combined_plot(self, state, score):
        # get rendered screen
        im_render = self.env.render(mode="rgb_array")
        n_stack = state.shape[-1]

        cmap = colors.ListedColormap(['red', 'skyblue', 'brown', 'blue'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        #obs_text = ['t (current frame)', 't-4', 't-8', 't-12']
        
        fig = plt.figure(dpi=100, figsize=(5.5, 4), constrained_layout=False, tight_layout=True)
        gs = fig.add_gridspec(4, 2, width_ratios=[4, 1])
        
        # individual obs frames
        for n in range(n_stack):
            ax = fig.add_subplot(gs[n, 1])
            im = ax.imshow(states[0,:,:,n], cmap=cmap, norm=norm)
            ax.set_axis_off()

        # rendered screen
        ax = fig.add_subplot(gs[:, 0])
        im = ax.imshow(im_render)
        ax.set_axis_off()
        ax.text(0, -5, 'score: '+str(int(score[0])))
        
        plt.show()
    
    def make_animation(self, deterministic=True, filename='gym_animation.gif', RETURN_FRAMES=False):
        '''
        Make an animation of the rendered screen
        '''
        # run policy
        frames = []
        states = self.env.reset()
        done = False
        
        while not done:
            #frames.append(self.env.render(mode="rgb_array"))
            im = self.env.render(mode="rgb_array")
            frames.append(im.copy())
            action, _ = self.model.predict(states, deterministic=deterministic)
            states, reward, done, info = self.env.step(action)
            
        if RETURN_FRAMES == False:
            # make animation
            imageio.mimsave(filename, frames, fps=50)
        else: # make animation manually in case Mario gets stuck in the level and drags the animation for too long
            return frames

    