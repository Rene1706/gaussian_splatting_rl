import torch
import math

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
    reward = min(max(psnr / 50.0, 0), 1)
    reward = reward / gaussians.num_points
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
        complexity_penalty = -rl_params.complexity_penalty * gaussian_count_change_normalized  # Removing Gaussians, potentially reward or reduce penalty
    
    #print(f"Complexity Penalty: {complexity_penalty}")
    # Adaptive reward scaling: increase reward for later iterations to handle diminishing PSNR changes
    if False:
        psnr_diff *= (1 + rl_params.late_reward_bonus * iteration)
    
    reward = psnr_diff - complexity_penalty
    #print("Reward: ", reward)
    return torch.tensor(reward, dtype=torch.float32, device="cuda")
