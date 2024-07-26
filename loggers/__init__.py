from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Union, Any, Dict
import torch
import wandb

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.loss_utils import l1_loss, ssim


@dataclass
class WandBInitConfig:
    project: str
    entity: Optional[str] = None
    name: Optional[str] = None
    id: Optional[str] = None
    job_type: Optional[str] = None
    dir: Optional[Path] = None
    reinit: Optional[bool] = None
    tags: Optional[Sequence] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    anonymous: Optional[str] = None
    mode: Optional[str] = None
    resume: Optional[Union[bool, str]] = None
    force: Optional[bool] = None
    save_code: Optional[bool] = None
    sync_tensorboard: Optional[bool] = None


class WandBLogger:
    def __init__(self, config: WandBInitConfig):
        wandb.init(**asdict(config))
        self.config = config
        self.image_interval = 200

    def log_train_iter_candidate(self, iteration, gaussians: GaussianModel, Ll1, ssim_value, loss, reward, image, gt_image):
        # Log these metrics every iteration
        wandb.log({
            'train_iter/l1_loss': Ll1.item(),
            'train_iter/loss': loss.item(),
            'train_iter/ssim': ssim_value.item(),
            'train_iter/reward': reward.item(),
        #    'train_iter/num_points': gaussians.num_points,
        }, step=iteration)
        
        # Log these metrics at intervals specified by self.image_interval
        if iteration % self.image_interval == 0:
            wandb.log({
                'train_iter/opacities': wandb.Histogram(gaussians.get_opacity.detach().cpu().numpy()),
                'train_iter/scaling_max': wandb.Histogram(gaussians.get_scaling.detach().max(dim=1).values.cpu().numpy()),
                'train_iter/gt_image': [wandb.Image(gt_image, caption="Ground Truth")],
                'train_iter/pred_image': [wandb.Image(image, caption="Prediction")]
            }, step=iteration)

    def log_densification_step(self, iteration, n_cloned, n_splitted, n_pruned, n_gaussians, n_noop):
        wandb.log({
            'densification_step/n_cloned': n_cloned,
            'densification_step/n_splitted': n_splitted,
            'densification_step/n_pruned': n_pruned,
            'densification_step/n_gaussians': n_gaussians,
            'densification_step/n_noop': n_noop,
        }, step=iteration)

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
