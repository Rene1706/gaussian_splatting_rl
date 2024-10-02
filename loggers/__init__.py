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


class WandBLogger:
    def __init__(self, wandb_args: WandbParams, wandb_config, last_iteration):
        filtered_wandb_args = {k: v for k, v in wandb_args.asdict().items() if v not in (None, '')}
        #print(filtered_wandb_args)
        wandb.init(**filtered_wandb_args, config=wandb_config)
        self.wandb_args = wandb_args
        self.image_interval = 5000
        self.last_iteration = last_iteration

    def log_train_iter_candidate(self, iteration, candidate_index, gaussians: GaussianModel, Ll1, psnr_value, ssim_value, loss, reward, image, gt_image, additional_rewards):
        iteration += self.last_iteration  # Adjust the iteration number
        if candidate_index == 0:
            log_data = {
                f'train_iter/candidate_{candidate_index}/l1_loss': Ll1.item(),
                f'train_iter/candidate_{candidate_index}/loss': loss.item(),
                f'train_iter/candidate_{candidate_index}/ssim': ssim_value.item(),
                f'train_iter/candidate_{candidate_index}/reward': reward,
                f'train_iter/candidate_{candidate_index}/psnr': psnr_value,
            }
        
        # Log additional rewards if provided
        if additional_rewards:
            for reward_name, reward_value in additional_rewards.items():
                log_data[f'train_iter/candidate_{candidate_index}/{reward_name}'] = reward_value
        if candidate_index == 0:
            wandb.log(log_data, step=iteration)
        
        # Log these metrics at intervals specified by self.image_interval
        if (iteration % self.image_interval == 0) and (candidate_index == 0):
            wandb.log({
                #f'train_iter/candidate_{candidate_index}/opacities': wandb.Histogram(gaussians.get_opacity.detach().cpu().numpy()),
                #f'train_iter/candidate_{candidate_index}/scaling_max': wandb.Histogram(gaussians.get_scaling.detach().max(dim=1).values.cpu().numpy()),
                f'train_iter/candidate_{candidate_index}/gt_image': [wandb.Image(gt_image, caption="Ground Truth")],
                f'train_iter/candidate_{candidate_index}/pred_image': [wandb.Image(image, caption="Prediction")]
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
        for i, adv in enumerate(advantage):
            wandb.log({f"rl_train_iter/candidate_{i}/advantage": adv.item()}, step = iteration)
