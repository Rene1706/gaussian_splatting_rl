#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, RLParams
import wandb
import importlib
from pathlib import Path
# Import the policy selector
from policies.action_selector import ParamBasedActionSelector
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, rlp, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    # Import the rewards module
    rewards_module = importlib.import_module("rewards.rewards")
    # Load reward functions
    reward_functions = [
        rewards_module.reward_psnr_normalized,
    ]
    # Get the reward function from the module
    reward_function = reward_functions[0]
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Initialize the action selector
    k = 1  # Number of candidates
    hidden_size = 16  # Adjust as needed
    action_selector = ParamBasedActionSelector(k=k, hidden_size=hidden_size).to("cuda")

    if rlp.base_model and Path(rlp.base_model).exists():
        print(f"Loading base_model from {rlp.base_model}")
        action_selector.param_network.load_state_dict(torch.load(rlp.base_model))

    # Load RL meta model, optimizer and scheduler
    if rlp.meta_model and Path(rlp.meta_model).exists():
        print(f"Loading meta_model from {rlp.meta_model}")
        action_selector.load_state_dict(torch.load(rlp.meta_model))

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        #print("Radii: ", radii)
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        with torch.no_grad():
            ssim_value = ssim(image, gt_image)
            psnr_value = psnr(image, gt_image)
            reward = reward_function(loss, psnr_value, gaussians)
            # Calculate and log additional rewards
            additional_rewards = {}
            #for func in reward_functions[1:]:
            #    additional_rewards[func.__name__] = func(loss, psnr(image, gt_image), gaussians)
            log_train_iter(iteration, gaussians, Ll1, psnr_value.mean().item(), ssim_value, loss, reward, image, gt_image, additional_rewards)
            if iteration % 1000 == 0:
                log_point_cloud(gaussians.point_cloud, iteration)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # Get actions from the policy selector
                    with torch.no_grad():
                        actions_list, inputs, log_probs = action_selector(
                            gaussians,
                            iteration=iteration,
                            scene_extent=scene.cameras_extent,
                        )
                    # Since k=1, we only have one set of actions
                    actions = actions_list[0]
                    # Apply actions to the gaussians
                    n_cloned, n_splitted, n_pruned, n_gaussians, n_noop = apply_actions(
                        gaussians,
                        actions,
                        min_opacity=0.005,
                        max_screen_size=size_threshold,
                        extent=scene.cameras_extent,
                    )
                    # Log input variables to densify_and_prune
                    log_data = {
                        "densify_grad_threshold": opt.densify_grad_threshold,
                        "value": 0.005,
                        "cameras_extent": scene.cameras_extent,
                        "size_threshold": size_threshold
                    }
                    with open(os.path.join(args.model_path, "densify_and_prune_log.txt"), "a") as log_file:
                        log_file.write(f"{log_data}\n")
                    wandb.log({
                        f'benchmark_densify/n_cloned': n_cloned,
                        f'benchmark_densify/n_splitted': n_splitted,
                        f'benchmark_densify/n_pruned': n_pruned,
                        f'benchmark_densify/n_gaussians': n_gaussians,
                        f'benchmark_densify/n_noop': n_noop,
                        f'benchmark_densify/% n_cloned': (n_cloned/n_gaussians)*100,
                        f'benchmark_densify/% n_splitted': (n_splitted/n_gaussians)*100,
                        f'benchmark_densify/% n_pruned': (n_pruned/n_gaussians)*100,
                        f'benchmark_densify/% n_noop': (n_noop/n_gaussians)*100
                    }, step=iteration)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def log_train_iter(iteration, gaussians: GaussianModel, Ll1, psnr_value, ssim_value, loss, reward, image, gt_image, additional_rewards):
        log_data = {
            f'train_iter/candidate_0/l1_loss': Ll1.item(),
            f'train_iter/candidate_0/loss': loss.item(),
            f'train_iter/candidate_0/ssim': ssim_value.item(),
            f'train_iter/candidate_0/num_points': gaussians.num_points,
            f'train_iter/candidate_0/reward': reward,
            f'train_iter/candidate_0/psnr': psnr_value,
        }
        
        # Log additional rewards if provided
        #if additional_rewards:
        #    for reward_name, reward_value in additional_rewards.items():
        #        log_data[f'train_iter/{reward_name}'] = reward_value

        #wandb.log(log_data, step=iteration)
        
        # Log these metrics at intervals specified by self.image_interval
        #if iteration % 1000 == 0:
        #    wandb.log({
                #f'train_iter/opacities': wandb.Histogram(gaussians.get_opacity.detach().cpu().numpy()),
                #f'train_iter/scaling_max': wandb.Histogram(gaussians.get_scaling.detach().max(dim=1).values.cpu().numpy()),
        #        f'train_iter/candidate_0/gt_image': [wandb.Image(gt_image, caption="Ground Truth")],
        #        f'train_iter/candidate_0/pred_image': [wandb.Image(image, caption="Prediction")]
        #    }, step=iteration)

def log_point_cloud(point_cloud, iteration=0):
        #wandb.log(
        #    {"point_cloud": wandb.Object3D(point_cloud)},
        #    step=iteration
        #)
        pass

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def apply_actions(gaussians: GaussianModel, actions: torch.Tensor, min_opacity, max_screen_size, extent):
    noop_mask = actions == 0
    clone_mask = actions == 1
    split_mask = actions == 2

    # Extend split mask to have the correct size after cloning
    n_cloned_points = torch.sum(clone_mask)
    split_mask = torch.cat(
        [
            split_mask,
            torch.zeros(n_cloned_points, device="cuda", dtype=torch.bool),
        ]
    )

    # Extend prune mask to have the correct size after cloning and splitting
    N = 2
    n_splitted_points = torch.sum(split_mask) * (N - 1)
    n_noop_points = torch.sum(noop_mask)

    # Number of points before densification is done for correct logging
    n_gaussians = gaussians.num_points

    # Clone and split
    gaussians.densify_and_clone_selected(clone_mask)
    gaussians.densify_and_split_selected(split_mask, N=N)

    # Prune points
    n_pruned_points = gaussians.select_and_prune_points(min_opacity, max_screen_size, extent)

    print(f"Cloned: {n_cloned_points}",
          f"Splitted: {n_splitted_points}",
          f"Pruned: {n_pruned_points}",
          f"NOOP: {torch.sum(noop_mask)}",
          f"NUMP: {n_gaussians}")
    
    torch.cuda.empty_cache()
    return n_cloned_points, n_splitted_points, n_pruned_points, n_gaussians, n_noop_points


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="master", mode="offline", save_code=True, tags=["default_3dgs"])
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    rlp = RLParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 100_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 100_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), rlp.extract(args),args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # Finish wandb run
    wandb.finish()
    # All done
    print("\nTraining complete.")
