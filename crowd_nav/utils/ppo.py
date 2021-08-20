import torch


def clipped_objective(
    log_pi: torch.Tensor,
    sampled_log_pi: torch.Tensor,
    advantages: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
    likelihood_ratio = torch.exp(log_pi - sampled_log_pi)

    # surrogate losses
    surr_loss_1 = likelihood_ratio * advantages
    surr_loss_2 = (
        torch.clamp(likelihood_ratio, min=1 - clip, max=1 + clip) * advantages
    )

    action_loss = -torch.min(surr_loss_1, surr_loss_2).mean()

    return action_loss


def clipped_value_objectives(
    value: torch.Tensor,
    sampled_value: torch.Tensor,
    sampled_return: torch.Tensor,
    clip: float,
):
    clipped_value = sampled_value + (value - sampled_value).clamp(
        min=-clip, max=clip
    )
    value_losses = (value - sampled_return).pow(2)
    value_losses_clipped = (clipped_value - sampled_return) ** 2
    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
    return value_loss
