"""
PPO Training
"""

from typing import Dict, List, Tuple, Callable, Union

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
import wandb
import tqdm
import os
import imageio

from game import Worker, Mario
from ppo import ClippedPPOLoss, ClippedValueFunctionLoss
from ppo.gae import GAE

from model import MlpPolicy

# Select device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device) 


class Trainer:
    """
    ## Trainer
    """

    def __init__(self, *,
                 policy_type: str, # 'cnn' or 'mlp'
                 env,
                 updates: int, 
                 epochs: int,
                 n_workers: int, 
                 worker_steps: int, 
                 batches: int,
                 value_loss_coef: float,
                 entropy_bonus_coef: float,
                 clip_range: float,
                 learning_rate: Union[float, Callable[[float], float]],
                 gamma: float,
                 lambda_: float,
                 use_wandb=False,
                 project_name="mario_ppo_training",
                 run_name="mario_ppo",
                 save_freq=0,
                ):
        # Initialize wandb if requested
        self.use_wandb = use_wandb
        self.run_name = run_name
        if self.use_wandb:
            # Handle learning rate display in wandb
            lr_value = learning_rate
            if callable(learning_rate):
                lr_value = learning_rate(1.0)  # Initial value
                
            wandb.init(project=project_name, name=run_name, config={
                "policy_type": policy_type,
                "updates": updates,
                "epochs": epochs,
                "n_workers": n_workers,
                "worker_steps": worker_steps,
                "batches": batches,
                "value_loss_coef": value_loss_coef,
                "entropy_bonus_coef": entropy_bonus_coef,
                "clip_range": clip_range,
                "learning_rate": lr_value,
                "gamma": gamma,
                "lambda": lambda_,
            })

        # #### Configurations

        # state shape
        self.env = env
        self.obs_shape = self.env.env.observation_space.shape # (13, 16, 4)
        self.obs_shape = [self.obs_shape[2], self.obs_shape[0], self.obs_shape[1]] # (4, 13, 16)
        self.action_size = self.env.env.action_space.n

        # number of updates
        self.updates = updates
        # number of epochs to train the model with sampled data
        self.epochs = epochs
        # number of worker processes
        self.n_workers = n_workers
        # number of steps to run on each process for a single update
        self.worker_steps = worker_steps
        # number of mini batches
        self.batches = batches
        # total number of samples for a single update
        self.batch_size = self.n_workers * self.worker_steps
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.batches
        assert (self.batch_size % self.batches == 0)

        # Value loss coefficient
        self.value_loss_coef = value_loss_coef
        # Entropy bonus coefficient
        self.entropy_bonus_coef = entropy_bonus_coef

        # Clipping range
        self.clip_range = clip_range
        # Learning rate
        self.learning_rate = learning_rate

        # #### Initialize

        # create workers, 47 is a random seed, each worker will have a different random seed
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]

        # initialize tensors for observations
        self.obs = np.zeros((self.n_workers, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]), dtype=np.uint8)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        # model
        if policy_type == 'mlp':
            self.model = MlpPolicy(self.obs_shape, self.action_size).to(device)
        else:
            raise ValueError(f"Invalid policy type: {policy_type}")

        # optimizer
        initial_lr = self.learning_rate
        if callable(self.learning_rate):
            initial_lr = self.learning_rate(1.0)  # Start with full learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_lr)

        # GAE with $\gamma = 0.99$ and $\lambda = 0.95$
        self.gae = GAE(self.n_workers, self.worker_steps, gamma, lambda_)

        # PPO Loss
        self.ppo_loss = ClippedPPOLoss()

        # Value Loss
        self.value_loss = ClippedValueFunctionLoss()

        # Save frequency
        self.save_freq = save_freq

    def sample(self) -> Dict[str, torch.Tensor]:
        """
        ### Sample data with current policy
        """
        #print("Sampling data with current policy")
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
        obs = np.zeros((self.n_workers, self.worker_steps, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]), dtype=np.uint8)
        log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps + 1), dtype=np.float32)

        with torch.no_grad():
            # sample `worker_steps` from each worker
            for t in range(self.worker_steps):
                #print(f"Sampling step {t} of {self.worker_steps}")
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs[:, t] = self.obs
                # sample actions from $\pi_{\theta_{OLD}}$ for each worker;
                #  this returns arrays of size `n_workers`
                pi, v = self.model(obs_to_torch(self.obs))
                values[:, t] = v.cpu().numpy()
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

                # run sampled actions on each worker
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", actions[w, t]))

                for w, worker in enumerate(self.workers):
                    # get results after executing the actions
                    self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()

                    # collect episode info, which is available if an episode finished;
                    #  this includes total reward and length of the episode -
                    #  look at `Game` to see how it works.
                    if info:
                        if self.use_wandb:
                            wandb.log({
                                "episode_reward": info['reward'],
                                "episode_length": info['length']
                            })

            # Get value of after the final step
            _, v = self.model(obs_to_torch(self.obs))
            values[:, self.worker_steps] = v.cpu().numpy()

        # calculate advantages
        advantages = self.gae(done, rewards, values)

        #
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values[:, :-1],
            'log_pis': log_pis,
            'advantages': advantages
        }

        # samples are currently in `[workers, time_step]` table,
        # we should flatten it for training
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat

    def train(self, samples: Dict[str, torch.Tensor]):
        """
        ### Train the model based on samples
        """
        # It learns faster with a higher number of epochs,
        #  but becomes a little unstable; that is,
        #  the average episode reward does not monotonically increase
        #  over time.
        # May be reducing the clipping range might solve it.
        num_batches = self.batches * self.epochs
        batch_pbar = tqdm.tqdm(total=num_batches, desc=f"Epoch {self.update_idx}/{self.updates}", leave=False)
        
        # Calculate progress for learning rate scheduler
        # Progress goes from 1 (beginning) to 0 (end)
        progress_remaining = 1.0 - (self.update_idx / self.updates)
        
        for _ in range(self.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # for each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                # train
                loss = self._calc_loss(mini_batch)

                # Set learning rate (using scheduler if provided)
                current_lr = self.learning_rate
                if callable(self.learning_rate):
                    current_lr = self.learning_rate(progress_remaining)
                
                for pg in self.optimizer.param_groups:
                    pg['lr'] = current_lr
                
                # Zero out the previously calculated gradients
                self.optimizer.zero_grad()
                # Calculate gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                # Update parameters based on gradients
                self.optimizer.step()
                
                # Update batch progress bar
                batch_pbar.update(1)
                batch_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.6f}"
                })

        # Close batch progress bar
        batch_pbar.close()
        
        # Log the current learning rate
        if self.use_wandb:
            lr_to_log = self.learning_rate
            if callable(self.learning_rate):
                lr_to_log = self.learning_rate(progress_remaining)
            wandb.log({"learning_rate": lr_to_log})
            
        print(f"Epoch [{self.update_idx}/{self.updates}], loss: {loss.item():.4f}, lr: {current_lr:.6f}")

    @staticmethod
    def _normalize(adv: torch.Tensor):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def _calc_loss(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        ### Calculate total loss
        The loss function is:
        $$
        \mathcal{L}(\theta) = \mathbb{E}\Bigl[ - \log \pi_\theta(a_t | s_t) \hat{R}_t \Bigr]
        $$
        where $\hat{R}_t$ is the reward-to-go.

        Args:
            samples: A dictionary containing the samples for training
                - 'obs': Observations
                - 'actions': Actions
                - 'log_pis': Log probabilities of actions
                - 'values': Values
                - 'advantages': Advantages

        Returns: loss
            A tensor containing the total loss
        """

        

        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        sampled_return = samples['values'] + samples['advantages']

        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        # where $\hat{A_t}$ is advantages sampled from $\pi_{\theta_{OLD}}$.
        # Refer to sampling function in [Main class](#main) below
        #  for the calculation of $\hat{A}_t$.
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        #  we are treating observations as state
        pi, value = self.model(samples['obs'])

        # TODO: [part d] Implement the PPO loss
        ### YOUR CODE HERE ###


        # Calculate policy loss policy_loss

        # Calculate Entropy Bonus entropy_bonus

        # Calculate value function loss

        pass
        ### END YOUR CODE ###

        # Calculate total loss
        # $\mathcal{L}^{CLIP+VF+EB} (\theta) =
        #  \mathcal{L}^{CLIP} (\theta) +
        #  c_1 \mathcal{L}^{VF} (\theta) - c_2 \mathcal{L}^{EB}(\theta)$
        loss = (policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_bonus_coef * entropy_bonus)

        # for monitoring
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()

        # Add to wandb logging
        if self.use_wandb:
            wandb.log({
                "policy_reward": -policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy_bonus": entropy_bonus.item(),
                "kl_div": approx_kl_divergence.item(),
                "clip_fraction": self.ppo_loss.clip_fraction
            })

        return loss

    def save_model(self, path):
        """
        Save model parameters to the specified path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_idx': self.update_idx,
        }, path)
        if self.use_wandb:
            wandb.save(path)

    def evaluate(self, n_episodes: int = 1, render: bool = True, deterministic: bool = True, speed: int = 2) -> Tuple[float, List[float]]:
        """
        Evaluate the current policy for n_episodes and optionally save a gif
        Returns:
            Tuple of (average reward, list of episode rewards)
        """
        self.model.eval()  # Set to evaluation mode
        episode_rewards = []
        episode_lengths = []
        frames = []
        
        # Create evaluation environment
        eval_env = Mario()  # Create a new environment for evaluation
        
        with torch.no_grad():
            for episode in range(n_episodes):
                obs = eval_env.reset()
                done = False
                episode_reward = 0.0
                max_steps = 10000
                step = 0
                while not done: # and step < max_steps:
                    step += 1
                    if render:
                        # Get RGB frame for gif
                        frame = eval_env.env.render(mode='rgb_array')
                        frames.append(frame.copy())
                    
                    # Convert observation to tensor and get action
                    obs_tensor = obs_to_torch(obs.reshape(1, *self.obs_shape))
                    pi, _ = self.model(obs_tensor)
                    if deterministic:
                        action = pi.probs.argmax(dim=-1).cpu().numpy()[0]
                    else:
                        action = pi.sample().cpu().numpy()[0]
                    
                    # Take step in environment
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(len(frames))
                
                if render and len(frames) > 0 and episode == 0:
                    # Save gif for this episode
                    path = f'results/{self.run_name}/eval_gifs'
                    os.makedirs(path, exist_ok=True)
                    gif_path = f'{path}/epoch_{self.update_idx}_ep{episode}.gif'
                    imageio.mimsave(gif_path, frames, fps=30)
                    frames = []  # Clear frames for next episode
                    
                    # if self.use_wandb:
                    #     wandb.log({
                    #         f"epoch_{self.update_idx}_ep{episode}.gif": wandb.Video(gif_path, format="gif"),
                    #     })
        
        # Close evaluation environment
        eval_env.env.close()
        
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        avg_length = sum(episode_lengths) / len(episode_lengths)
        if self.use_wandb:
            wandb.log({
                "eval_average_reward": avg_reward,
                "eval_rewards": episode_rewards,
                "eval_lengths": avg_length
            })
        
        self.model.train()  # Set back to training mode
        return avg_reward, avg_length

    def run_training_loop(self):
        """
        ### Run training loop
        """
        self.update_idx = 0
        for update in range(self.updates):
            self.update_idx += 1
            # sample with current policy
            samples = self.sample()

            # train the model
            self.train(samples)


            if self.use_wandb:
                wandb.log({"update": update})

            # Save model and evaluate if save_freq is set and we've hit the interval
            if self.save_freq > 0 and (update + 1) % self.save_freq == 0:
                save_path = f"results/{self.run_name}/checkpoints/model_checkpoint_epoch_{update + 1}.pt"
                self.save_model(save_path)
                
                # Run evaluation
                avg_reward, avg_length = self.evaluate(n_episodes=4, render=True, deterministic=True)
                print(f"Completed update {update + 1} - Saved model to {save_path} - Evaluation reward: {avg_reward:.2f} - Evaluation length: {avg_length:.2f}")

    def destroy(self):
        """
        ### Destroy
        Stop the workers and cleanup wandb
        """
        for worker in self.workers:
            worker.child.send(("close", None))
        
        if self.use_wandb:
            wandb.finish()


def main():
    env = Mario()
    # Configurations
    configs = {
        # Policy type
        'policy_type': 'mlp',
        # Environment
        'env': env,
        # Number of updates
        'updates': 3,  # Increased for better training
        # Number of epochs to train the model with sampled data
        'epochs': 2,
        # Number of worker processes
        'n_workers': 1,
        # Number of steps to run on each process for a single update
        'worker_steps': 128,  # Increased for better training
        # Number of mini batches
        'batches': 4,
        # Value loss coefficient
        'value_loss_coef': 0.5,
        # Entropy bonus coefficient
        'entropy_bonus_coef': 0.01,
        # GAE lambda
        'lambda_': 0.95,
        # Gamma
        'gamma': 0.99,
        # Clip range
        'clip_range': 0.1,
        # Learning rate
        'learning_rate': 1e-3,
        # Use wandb for logging
        'use_wandb': False,
        'project_name': 'ppo_training',
        'run_name': 'ppo_experiment_gif',
        # Save and evaluate model every 2 updates
        'save_freq': 1,
    }

    # Initialize the trainer
    trainer = Trainer(**configs)
    
    # Run training loop
    trainer.run_training_loop()
    
    # Final evaluation with multiple episodes
    print("\nRunning final evaluation...")
    avg_reward, episode_rewards = trainer.evaluate(n_episodes=1, render=True, deterministic=True)
    print(f"Final evaluation - Average reward: {avg_reward:.2f}")
    print(f"Episode rewards: {episode_rewards}")
    
    trainer.destroy()


# ## Run it
if __name__ == "__main__":
    main()
