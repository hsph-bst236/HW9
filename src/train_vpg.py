"""
---
title: Vanilla Policy Gradient Implementation
summary: Implementation of vanilla policy gradient for reinforcement learning
---

# Vanilla Policy Gradient

This implementation inherits from the PPO Trainer but replaces GAE with reward-to-go
and only uses policy gradient loss without value function estimation.
"""

from typing import Dict, Callable, Union
import numpy as np
import torch
from torch import nn
import wandb
import tqdm

# Update imports to be relative to project structure
from train_ppo import Trainer, obs_to_torch
from ppo.gae import reward_to_go

class VPGTrainer(Trainer):
    """
    ## Vanilla Policy Gradient Trainer
    
    This trainer inherits from the PPO Trainer but uses reward-to-go instead of GAE
    and removes value function estimation.
    """
    
    def __init__(self, **kwargs):
        # Call the parent class's __init__ method with the same arguments
        super().__init__(**kwargs)
        # Remove GAE (we'll use reward-to-go directly in sample method)
        delattr(self, 'gae')
        # Store gamma for reward-to-go calculation
        self.gamma = kwargs.get('gamma', 0.99)
        
    def sample(self) -> Dict[str, torch.Tensor]:
        """
        ### Sample data with current policy
        
        Similar to PPO's sample method but uses reward-to-go instead of GAE
        """
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
        obs = np.zeros((self.n_workers, self.worker_steps, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]), dtype=np.uint8)
        log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        
        with torch.no_grad():
            # sample `worker_steps` from each worker
            for t in range(self.worker_steps):
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs[:, t] = self.obs
                # sample actions from current policy for each worker
                pi, _ = self.model(obs_to_torch(self.obs))
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

                # run sampled actions on each worker
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", actions[w, t]))

                for w, worker in enumerate(self.workers):
                    # get results after executing the actions
                    self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()

                    # log episode info if available
                    if info:
                        if self.use_wandb:
                            wandb.log({
                                "episode_reward": info['reward'],
                                "episode_length": info['length']
                            })

        # calculate rewards-to-go instead of advantages
        returns = reward_to_go(done, rewards, self.gamma)

        samples = {
            'obs': obs,
            'actions': actions,
            'log_pis': log_pis,
            'returns': returns  # Using returns instead of advantages
        }

        # Flatten samples for training
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

        return samples_flat
    
    def _calc_loss(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        ### Calculate policy gradient loss
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
                - 'returns': Reward-to-go

        Returns: policy_loss
            A tensor containing the policy gradient loss
        """
        # TODO: [part c] Implement the policy gradient loss
        ### YOUR CODE HERE ###
        pass

        # Log metrics
        if self.use_wandb:
            wandb.log({
                "policy_reward": 0,  # TODO: [part c] log the policy reward -log_pi * returns
                "entropy_bonus": 0   # TODO: [part c] log the entropy of pi
            })
        ### END YOUR CODE ###
                    
    def train(self, samples: Dict[str, torch.Tensor]):
        """
        ### Train the model based on samples
        
        Override parent class's train method to support learning rate scheduler
        """
        # Calculate number of mini-batches
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

def main():
    from train_ppo import Mario
    env = Mario()
    
    # Configurations
    configs = {
        # Policy type
        'policy_type': 'mlp',
        # Environment
        'env': env,
        # Number of updates
        'updates': 3,
        # Number of epochs to train the model with sampled data
        'epochs': 2,
        # Number of worker processes
        'n_workers': 1,
        # Number of steps to run on each process for a single update
        'worker_steps': 128,
        # Number of mini batches
        'batches': 4,
        # Gamma discount factor
        'gamma': 0.99,
        # Learning rate
        'learning_rate': 1e-3,
        # Use wandb for logging
        'use_wandb': False,
        'project_name': 'vpg_training',
        'run_name': 'vpg_experiment',
        # Save and evaluate model every update
        'save_freq': 1,
    }

    # Initialize the VPG trainer
    trainer = VPGTrainer(**configs)
    
    # Run training loop
    trainer.run_training_loop()
    
    trainer.destroy()

if __name__ == "__main__":
    main()