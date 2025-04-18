"""
Run training using Vanilla Policy Gradient
"""

from game import Mario
from train_vpg import VPGTrainer
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def main():
    """Main function to run the Vanilla Policy Gradient training"""
    env = Mario()
    # Configurations
    configs = {
        # Policy type
        'policy_type': 'mlp',
        # Environment
        'env': env,
        # Number of updates
        'updates': 10_000,  
        # Number of epochs to train the model with sampled data
        'epochs': 10,
        # Number of worker processes
        'n_workers': 8,
        # Number of steps to run on each process for a single update
        'worker_steps': 2048,  
        # Number of mini batches
        'batches': 128,
        # Value loss coefficient (required by parent class but not used in VPG)
        'value_loss_coef': 0.5,
        # Entropy bonus coefficient
        'entropy_bonus_coef': 0.0,
        # GAE lambda (required by parent class but not used in VPG)
        'lambda_': 0.95,
        # Gamma
        'gamma': 0.99,
        # Clip range (required by parent class but not used in VPG)
        'clip_range': 0.2,
        # You could also try a linear scheduler 
        'learning_rate': 3e-4,  # Or linear_schedule(3e-4)
        # Use wandb for logging
        'use_wandb': True,
        'project_name': 'hw9_mario',
        'run_name': 'vpg_mlp',
        # Save and evaluate model
        'save_freq': 50,
    }

    # Initialize the trainer
    trainer = VPGTrainer(**configs)

    try:
        # Run training loop
        trainer.run_training_loop()
    finally:
        # Ensure resources are properly cleaned up
        trainer.destroy()

if __name__ == '__main__':
    main()