from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Union, Any, Dict
import torch
import wandb

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.loss_utils import l1_loss, ssim
from arguments import WandbParams
import numpy as np


class WandBLogger:
    def __init__(self, wandb_args: WandbParams, wandb_config, last_iteration):
        filtered_wandb_args = {k: v for k, v in wandb_args.asdict().items() if v not in (None, '')}
        #print(filtered_wandb_args)
        wandb.init(**filtered_wandb_args, config=wandb_config)
        self.wandb_args = wandb_args
        self.image_interval = 5000
        self.last_iteration = last_iteration

    def log_optimization_iteration(self, iteration, candidate_index, gaussians: GaussianModel, Ll1, psnr_value, ssim_value, loss, image, gt_image):
        iteration += self.last_iteration  # Adjust the iteration number
        if candidate_index == 0:
            log_data = {
                f'optimization_iter/candidate_{candidate_index}/l1_loss': Ll1.item(),
                f'optimization_iter/candidate_{candidate_index}/loss': loss.item(),
                f'optimization_iter/candidate_{candidate_index}/ssim': ssim_value.item(),
                f'optimization_iter/candidate_{candidate_index}/psnr': psnr_value,
            }
            wandb.log(log_data, step=iteration)

            # Log images at intervals specified by self.image_interval
            if (iteration % self.image_interval == 0):
                wandb.log({
                    f'optimization_iter/candidate_{candidate_index}/gt_image': [wandb.Image(gt_image, caption="Ground Truth")],
                    f'optimization_iter/candidate_{candidate_index}/pred_image': [wandb.Image(image, caption="Prediction")]
                }, step=iteration)

    def log_densification_step(self, iteration, candidate_index, n_cloned, n_splitted, n_pruned, n_gaussians, n_noop):
        iteration += self.last_iteration  # Adjust the iteration number
        if candidate_index == 0:
            wandb.log({
                f'densification_step/candidate_{candidate_index}/n_cloned': n_cloned,
                f'densification_step/candidate_{candidate_index}/n_splitted': n_splitted,
                f'densification_step/candidate_{candidate_index}/n_pruned': n_pruned,
                f'densification_step/candidate_{candidate_index}/n_gaussians': n_gaussians,
                f'densification_step/candidate_{candidate_index}/n_noop': n_noop,
                f'densification_step/candidate_{candidate_index}/% n_cloned': (n_cloned/n_gaussians)*100,
                f'densification_step/candidate_{candidate_index}/% n_splitted': (n_splitted/n_gaussians)*100,
                f'densification_step/candidate_{candidate_index}/% n_pruned': (n_pruned/n_gaussians)*100,
                f'densification_step/candidate_{candidate_index}/% n_noop': (n_noop/n_gaussians)*100
            }, step=iteration)

    def log_point_cloud(self, point_cloud, iteration=0):
        iteration += self.last_iteration  # Adjust the iteration number
        wandb.log(
            {"point_cloud": wandb.Object3D(point_cloud)},
            step=iteration
        )

    def log_rl_loss(self, iteration, loss, advantage, policy_optimizer):
        iteration += self.last_iteration  # Adjust the iteration number
        lr = policy_optimizer.param_groups[0]['lr']
        wandb.log({
            "rl_train_iter/learning_rate": lr,
            "rl_train_iter/loss": loss.item(),
        },step = iteration)

    def log_densification_iteration(self, iteration, candidate_index, reward, per_gaussian_rewards, additional_rewards=None):
        iteration += self.last_iteration  # Adjust the iteration number
        if candidate_index == 0:
            log_data = {
                f'densification_iter/candidate_{candidate_index}/reward': reward,
            }

            # Log additional rewards if provided
            if additional_rewards:
                for reward_name, reward_value in additional_rewards.items():
                    log_data[f'densification_iter/candidate_{candidate_index}/{reward_name}'] = reward_value

            # Ensure per_gaussian_rewards is a NumPy array
            if isinstance(per_gaussian_rewards, torch.Tensor):
                per_gaussian_rewards_np = per_gaussian_rewards.detach().cpu().numpy()
            else:
                per_gaussian_rewards_np = np.array(per_gaussian_rewards)

            # Compute statistical summaries
            mean_reward = np.mean(per_gaussian_rewards_np)
            median_reward = np.median(per_gaussian_rewards_np)
            std_reward = np.std(per_gaussian_rewards_np)
            max_reward = np.max(per_gaussian_rewards_np)
            min_reward = np.min(per_gaussian_rewards_np)

            # Prepare data for logging
            log_data.update({
                f'densification_iter/candidate_{candidate_index}/per_gaussian_rewards/mean': mean_reward,
                f'densification_iter/candidate_{candidate_index}/per_gaussian_rewards/median': median_reward,
                f'densification_iter/candidate_{candidate_index}/per_gaussian_rewards/std': std_reward,
                f'densification_iter/candidate_{candidate_index}/per_gaussian_rewards/max': max_reward,
                f'densification_iter/candidate_{candidate_index}/per_gaussian_rewards/min': min_reward,
                f'densification_iter/candidate_{candidate_index}/per_gaussian_rewards/histogram': wandb.Histogram(per_gaussian_rewards_np)
            })

            wandb.log(log_data, step=iteration)