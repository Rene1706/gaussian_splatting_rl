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
    def __init__(self, config: WandbParams):
        filtered_config = {k: v for k, v in config.asdict().items() if v not in (None, '')}
        #print(filtered_config)
        wandb.init(**filtered_config)
        self.config = config
        self.image_interval = 200

    def log_train_iter_candidate(self, iteration, candidate_index, gaussians: GaussianModel, Ll1, ssim_value, loss, reward, image, gt_image):
        # Log these metrics every iteration
        wandb.log({
            f'train_iter/candidate_{candidate_index}/l1_loss': Ll1.item(),
            f'train_iter/candidate_{candidate_index}/loss': loss.item(),
            f'train_iter/candidate_{candidate_index}/ssim': ssim_value.item(),
            f'train_iter/candidate_{candidate_index}/reward': reward,
        #    f'train_iter/candidate_{candidate_index}num_points': gaussians.num_points,
        }, step=iteration)
        
        # Log these metrics at intervals specified by self.image_interval
        if iteration % self.image_interval == 0:
            wandb.log({
                f'train_iter/candidate_{candidate_index}/opacities': wandb.Histogram(gaussians.get_opacity.detach().cpu().numpy()),
                f'train_iter/candidate_{candidate_index}/scaling_max': wandb.Histogram(gaussians.get_scaling.detach().max(dim=1).values.cpu().numpy()),
                f'train_iter/candidate_{candidate_index}/gt_image': [wandb.Image(gt_image, caption="Ground Truth")],
                f'train_iter/candidate_{candidate_index}/pred_image': [wandb.Image(image, caption="Prediction")]
            }, step=iteration)

    def log_densification_step(self, iteration, candidate_index, n_cloned, n_splitted, n_pruned, n_gaussians, n_noop):
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
        wandb.log(
            {"point_cloud": wandb.Object3D(point_cloud)},
            step=iteration
        )

    def log_rl_loss(self, iteration, loss, advantage, policy_optimizer):
        lr = policy_optimizer.param_groups[0]['lr']
        wandb.log({
            "rl_train_iter/learning_rate": lr,
            "rl_train_iter/loss": loss.item(),
        },step = iteration)
        for i, adv in enumerate(advantage):
            wandb.log({f"rl_train_iter/candidate_{i}/advantage": adv.item()}, step = iteration)

    def log_evaluation(self, iteration, gaussians: GaussianModel, scene: Scene, renderFunc, renderArgs: tuple):
        torch.cuda.empty_cache()
        validation_configs = [
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "train", "cameras": scene.getTrainCameras()},
        ]
        for config in validation_configs:
            n_cameras = len(config["cameras"])
            if n_cameras > 0:
                l1_total, psnr_total, ssim_total = 0.0, 0.0, 0.0
                images, gt_images = [], []
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images.append(image)
                    gt_images.append(gt_image)
                    l1_total += l1_loss(image, gt_image).mean().double()
                    psnr_total += psnr(image, gt_image).mean().double()
                    ssim_total += ssim(image, gt_image).mean().double()
                l1_avg = l1_total / n_cameras
                psnr_avg = psnr_total / n_cameras
                ssim_avg = ssim_total / n_cameras

                wandb.log({
                    f"{config['name']}/psnr": psnr_avg,
                    f"{config['name']}/l1": l1_avg,
                    f"{config['name']}/ssim": ssim_avg,
                    f"{config['name']}/images": [wandb.Image(img, caption=f"Rendered {config['name']} Image {idx}") for idx, img in enumerate(images)],
                    f"{config['name']}/gt_images": [wandb.Image(img, caption=f"GT {config['name']} Image {idx}") for idx, img in enumerate(gt_images)]
                }, step=iteration)
        torch.cuda.empty_cache()
