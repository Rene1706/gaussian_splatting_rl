import torch
import math

def reward_default(loss, psnr_value, gaussians):
    penalty_factor = 10
    return -loss.detach() - penalty_factor * gaussians.num_points

def reward_function_2(loss, psnr_value, gaussians):
    penalty_factor = 5
    return -loss.detach() - penalty_factor * math.log(gaussians.num_points)

def reward_function_3(loss, psnr_value, gaussians):
    penalty_factor = 0.001
    return -loss.detach()/gaussians.num_points - penalty_factor * math.log(gaussians.num_points)
    
def reward_function_4(loss, psnr_value, gaussians):
    return -loss.detach()/gaussians.num_points

def reward_function_5(loss, psnr_value, gaussians):
    num_points = gaussians.num_points
    loss_impact = 10 * loss.detach()
    
    return -loss_impact/num_points

def reward_psnr_normalized(loss, psnr, gaussians):
    # Normalize PSNR to the range [0, 1]
    if isinstance(psnr, torch.Tensor):
        # Reduce the tensor to a scalar value by taking the mean
        psnr = psnr.mean().item()

    # Normalize PSNR to the range [0, 1]
    reward = min(max(psnr / 50.0, 0), 1)
    return torch.tensor(reward, device="cuda")

def reward_psnr_normalized_2(loss, psnr, gaussians):
    # Normalize PSNR to the range [0, 1]
    if isinstance(psnr, torch.Tensor):
        # Reduce the tensor to a scalar value by taking the mean
        psnr = psnr.mean().item()

    # Normalize PSNR to the range [0, 1]
    reward = min(max(psnr / 50.0, 0), 1)
    reward = reward / gaussians.num_points
    return torch.tensor(reward, device="cuda")

def reward_psnr_normalized_3(loss, psnr, gaussians):
    # Normalize PSNR to the range [0, 1]
    if isinstance(psnr, torch.Tensor):
        # Reduce the tensor to a scalar value by taking the mean
        psnr = psnr.mean().item()

    # Normalize PSNR to the range [0, 1]
    psnr_normalized = min(max(psnr / 40.0, 0), 1)

    # Define weights for PSNR and the number of Gaussians
    psnr_weight = 0.9  # Emphasize PSNR more
    gaussians_weight = 0.1  # But still consider the number of Gaussians

    # Calculate reward
    reward = psnr_weight * psnr_normalized - gaussians_weight * (gaussians.num_points / 10000.0)
    reward = max(reward, 0)  # Ensure reward is non-negative
    return torch.tensor(reward, dtype=torch.float32,device="cuda")
    