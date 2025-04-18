"""
A PyTorch implementation/tutorial of Generalized Advantage Estimation (GAE).
"""

import numpy as np


def reward_to_go(done: np.ndarray, rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Calculate the reward-to-go for each state.
    
    Reward-to-go is the discounted sum of future rewards from each state.
    It's computed as:
    
    $R_t = \sum_{i=t}^{T} \gamma^{i-t} r_i$
    
    Args:
        done: Boolean array with shape (n_workers, worker_steps) indicating if an episode is done
        rewards: Array with shape (n_workers, worker_steps) containing the rewards
        gamma: Discount factor
        
    Returns:
        Array with shape (n_workers, worker_steps) containing the rewards-to-go
    """
    n_workers, worker_steps = rewards.shape
    returns = np.zeros_like(rewards, dtype=np.float32)
    
    # TODO: [part b] Implement the reward-to-go calculation
    ### YOUR CODE HERE ###
    pass
    ### END YOUR CODE ###

class GAE:
    def __init__(self, n_workers: int, worker_steps: int, gamma: float, lambda_: float):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.worker_steps = worker_steps
        self.n_workers = n_workers

    def __call__(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Calculate advantages

        Args:
            done: Boolean array with shape (n_workers, worker_steps) indicating if an episode is done
            rewards: Array with shape (n_workers, worker_steps) containing the rewards
            values: Array with shape (n_workers, worker_steps) containing the values
            
        Returns:
            Array with shape (n_workers, worker_steps) containing the advantages
        """

        # advantages table
        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        
        # TODO: [part b] Implement the GAE calculation
        ### YOUR CODE HERE ###
        pass
        ### END YOUR CODE ###

        # $\hat{A_t}$
        return advantages


