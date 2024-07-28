import torch
import math

def reward_default(loss, gaussians):
    penalty_factor = 10
    return -loss.detach() - penalty_factor * gaussians.num_points

def reward_function_2(loss, gaussians):
    penalty_factor = 5
    return -loss.detach() - penalty_factor * math.log(gaussians.num_points)

def reward_function_3(loss, gaussians):
    penalty_factor = 0.001
    return -loss.detach()/gaussians.num_points - penalty_factor * math.log(gaussians.num_points)
    
def reward_function_4(loss, gaussians):
    return -loss.detach()/gaussians.num_points

def reward_function_5(loss, gaussians):
    num_points = gaussians.num_points
    loss_impact = 10 * loss.detach()
    
    return -loss_impact/num_points