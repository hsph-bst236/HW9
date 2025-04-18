import torch
from torch import nn


class ClippedPPOLoss(nn.Module):
    """
    ## PPO Loss

    Here's how the PPO update rule is derived.

    Args:
        log_pi: The log of the policy.
        sampled_log_pi: The log of the sampled policy.
        advantage: The advantage.
        clip: The clip value.

    Returns:
        The PPO loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, log_pi: torch.Tensor, sampled_log_pi: torch.Tensor,
                advantage: torch.Tensor, clip: float) -> torch.Tensor:
        # TODO: [part d] Implement the PPO loss
        ### START CODE HERE ###
        pass
        ### END CODE HERE ###


class ClippedValueFunctionLoss(nn.Module):
    """
    ## Least Square Value Function Loss
    We named it as Clipped Value Function Loss for furture generalization.
    Args:
        value: The value.
        sampled_return: The sampled return.

    Returns:
        The least square value function loss.
    """

    def forward(self, value: torch.Tensor, sampled_return: torch.Tensor):
        vf_loss = (value - sampled_return) ** 2
        return 0.5 * vf_loss.mean()
