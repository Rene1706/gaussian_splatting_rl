import torch
import math
import numpy as np

def reward_default(**kwargs):
    loss = kwargs.get('loss')
    gaussians = kwargs.get('gaussians')
    penalty_factor = 10
    return -loss.detach() - penalty_factor * gaussians.num_points

def reward_function_2(**kwargs):
    loss = kwargs.get('loss')
    gaussians = kwargs.get('gaussians')
    penalty_factor = 5
    return -loss.detach() - penalty_factor * math.log(gaussians.num_points)

def reward_function_3(**kwargs):
    loss = kwargs.get('loss')
    gaussians = kwargs.get('gaussians')
    penalty_factor = 0.001
    return -loss.detach() / gaussians.num_points - penalty_factor * math.log(gaussians.num_points)

def reward_function_4(**kwargs):
    loss = kwargs.get('loss')
    gaussians = kwargs.get('gaussians')
    return -loss.detach() / gaussians.num_points

def reward_function_5(**kwargs):
    loss = kwargs.get('loss')
    gaussians = kwargs.get('gaussians')
    num_points = gaussians.num_points
    loss_impact = 10 * loss.detach()
    return -loss_impact / num_points

def reward_function_6(**kwargs):
    loss = kwargs.get('loss')
    gaussians = kwargs.get('gaussians')
    return -loss.detach() / math.log(gaussians.num_points)

def reward_psnr_normalized(**kwargs):
    psnr = kwargs.get('psnr')
    if isinstance(psnr, torch.Tensor):
        psnr = psnr.mean().item()
    reward = min(max(psnr / 50.0, 0), 1)
    return torch.tensor(reward, device="cuda")

def reward_psnr_normalized_2(**kwargs):
    psnr = kwargs.get('psnr')
    gaussians = kwargs.get('gaussians')
    if isinstance(psnr, torch.Tensor):
        psnr = psnr.mean().item()

    # Normalize PSNR to the range [0, 1]
    reward = min(max(psnr / 45.0, 0), 1)

    # Check that the number of Gaussians is not zero to avoid division by zero
    if gaussians.num_points >= 1:
        reward = reward / math.log(gaussians.num_points)
    else:
        # Handle the case where num_points is zero (e.g., return a minimal reward)
        reward = -10.0

    return torch.tensor(reward, device="cuda")

def reward_psnr_normalized_log_num_gauss(**kwargs):
    psnr = kwargs.get('psnr')
    gaussians = kwargs.get('gaussians')
    if isinstance(psnr, torch.Tensor):
        psnr = psnr.mean().item()
    reward = min(max(psnr / 45.0, 10), 1)
    if gaussians.num_points <= 1:
        reward = -10.0
    else:
        reward = reward / math.log(gaussians.num_points)
    return torch.tensor(reward, device="cuda")

def reward_psnr_normalized_3(**kwargs):
    psnr = kwargs.get('psnr')
    gaussians = kwargs.get('gaussians')
    if isinstance(psnr, torch.Tensor):
        psnr = psnr.mean().item()
    psnr_normalized = min(max(psnr / 40.0, 0), 1)
    psnr_weight = 0.9
    gaussians_weight = 0.1
    reward = psnr_weight * psnr_normalized - gaussians_weight * (gaussians.num_points / 10000.0)
    reward = max(reward, 0)
    return torch.tensor(reward, dtype=torch.float32, device="cuda")

def reward_diff_psnr(**kwargs):
    # Get the arguments
    psnr = kwargs.get('psnr')
    gaussians = kwargs.get('gaussians')
    last_psnr = kwargs.get('last_psnr')
    delta_gaussians = kwargs.get('delta_gaussians')
    gaussians = kwargs.get('gaussians')
    rl_params = kwargs.get('rl_params')
    iteration = kwargs.get('iteration')
    # Calculate the reward
    psnr_diff = psnr.mean().item() - last_psnr
    #print(f"PSNR: {psnr.mean().item()}, Last PSNR: {last_psnr}, PSNR Diff: {psnr_diff}")
    # Complexity penalty
    # Normalize or log-scale the Gaussian count change to prevent excessive penalties
    # Use absolute value for logarithmic scaling to avoid math domain errors
    gaussian_count_change_normalized = math.log(1 + abs(delta_gaussians))

    # Apply penalty or bonus based on whether gaussians were added or removed
    if delta_gaussians > 0:
        complexity_penalty = rl_params.complexity_penalty * gaussian_count_change_normalized  # Adding Gaussians, penalize
    else:
        # Setting to 0 to avoid big reward when pruning is done
        complexity_penalty = 0  # Removing Gaussians, potentially reward or reduce penalty
    
    #print(f"Complexity Penalty: {complexity_penalty}")
    # Adaptive reward scaling: increase reward for later iterations to handle diminishing PSNR changes
    if False:
        psnr_diff *= (1 + rl_params.late_reward_bonus * iteration)
    
    reward = (rl_params.psnr_weight * psnr_diff) - complexity_penalty
    #print("Reward: ", reward)
    return torch.tensor(reward, dtype=torch.float32, device="cuda")

def reward_diff_psnr_relative(**kwargs):
    # Get the arguments
    psnr = kwargs.get('psnr')
    gaussians = kwargs.get('gaussians')
    last_psnr = kwargs.get('last_psnr')
    delta_gaussians = kwargs.get('delta_gaussians')
    gaussians = kwargs.get('gaussians')
    rl_params = kwargs.get('rl_params')
    iteration = kwargs.get('iteration')
    # Calculate the reward
    psnr_diff = psnr - last_psnr
    #print(f"PSNR: {psnr.mean().item()}, Last PSNR: {last_psnr}, PSNR Diff: {psnr_diff}")
    # Complexity penalty
    relative_delta_gaussians = delta_gaussians / gaussians.num_points

    # Multiply by 10 to bring into same scale as reward_diff_psnr for sweep
    relative_delta_gaussians = relative_delta_gaussians * 100

    # Apply penalty or bonus based on whether gaussians were added or removed
    if delta_gaussians > 0:
        complexity_penalty = rl_params.complexity_penalty * relative_delta_gaussians  # Adding Gaussians, penalize
    else:
        complexity_penalty = 0  # Removing Gaussians, potentially reward or reduce penalty
    
    #print(f"Complexity Penalty: {complexity_penalty}")
    # Adaptive reward scaling: increase reward for later iterations to handle diminishing PSNR changes
    if False:
        psnr_diff *= (1 + rl_params.late_reward_bonus * iteration)
    
    reward = (rl_params.psnr_weight * psnr_diff) - complexity_penalty
    #print("Reward: ", reward)
    return torch.tensor(reward, dtype=torch.float32, device="cuda")

def reward_pareto(**kwargs):
    # Get the arguments
    psnr = kwargs.get('psnr')
    gaussians = kwargs.get('gaussians')
    current_point = [gaussians.num_points, psnr]
    # Load the fitted curve data
    data = np.load("/bigwork/nhmlhuer/git/master_evaluation/fitted_curve_data.npz")
    x_fit = data['x_fit']
    y_fit = data['y_fit']
    
    # Find the PSNR on the fitted curve that corresponds to the current number of Gaussians
    psnr_curve = np.interp(gaussians.num_points, x_fit, y_fit)
    
    # Calculate the PSNR distance (difference between current PSNR and the curve PSNR)
    psnr_distance = np.abs(psnr - psnr_curve)
    
    # Define a simple reward function based on the PSNR distance
    def calculate_reward(distance, scale_factor=1):
        # Cap the reward so that very small distances don't give excessively high rewards
        reward = scale_factor / (1 + distance)
        return reward
    
    # Calculate the reward based on PSNR distance
    reward = calculate_reward(psnr_distance)
    
    # Return the reward as a tensor
    return torch.tensor(reward, dtype=torch.float32, device="cuda")

