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
import sys
import datetime
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
from random import randint

import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import csv
# Rene: Added for configs as not yet using new configs
from arguments import ModelParams, PipelineParams, OptimizationParams, WandbParams, RLParams

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

from gaussian_renderer import network_gui, render
from scene import Scene, GaussianModel
from policies.action_selector import ParamBasedActionSelector
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
import seaborn as sns
from loggers import WandBLogger
import importlib

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

DO_PLOT = False
CSV_LOG_FILE = "output/pareto.csv"  # if None -> Dont log psnr, ... in file


def visualize_grad_scaling(gaussians, name, scene, *, actions=None):
    extent = scene.cameras_extent
    with torch.no_grad():
        grads = gaussians.xyz_gradient_accum / gaussians.denom
        grads[grads.isnan()] = 0.0
        plot_data_path = Path(scene.model_path) / "plot_data" / f"grad_scaling_{name}.torch"
        plot_data_path.parent.mkdir(exist_ok=True, parents=True)
        data = {
            "grad_norms": torch.norm(grads, dim=-1),
            "scalings": gaussians._scaling,#torch.max(gaussians.scalings, dim=1).values,
            "scene_extent": scene.cameras_extent,
            "name": name,
            "gaussians_percent_dense": gaussians.percent_dense
        }
        if actions is not None:
            data["actions"] = actions
        torch.save(data, plot_data_path)

    if not DO_PLOT:
        return

    # Convert to plottable numpy
    data = {k: v.detach().cpu().numpy() for k, v in data.items()}

    g = sns.JointGrid(data, x="grad_norms", y="scalings", marginal_ticks=True, xlim=[10 ** -10, 1],
                      ylim=[10 ** -4, 10 ** 2])
    g.ax_joint.set(yscale="log", xscale="log")
    g.ax_marg_x.set(title=name)
    cax = g.figure.add_axes([.15, .55, .02, .2])
    g.plot_joint(
        sns.histplot,
        cbar=True, cbar_ax=cax
    )
    grad_norm_threshold = 0.0002
    scaling_threshold = gaussians.percent_dense * extent
    g.ax_joint.axvline(x=grad_norm_threshold)
    g.ax_joint.hlines(y=scaling_threshold, xmin=grad_norm_threshold, xmax=1)
    g.ax_joint.text(1, scaling_threshold + 0.01, "Split", ha='right')
    g.ax_joint.text(1, scaling_threshold - 0.02, "Clone", ha='right')
    g.plot_marginals(sns.histplot, element="step")
    plt_folder = Path(scene.model_path) / "grad_scale_plots"
    plt_folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(plt_folder / f"{name}.jpg")
    plt.close()


def save_to_log(scene_name, iteration, psnr, l1, num_gaussians, model_folder):
    if CSV_LOG_FILE is None:
        return

    file_path = Path(CSV_LOG_FILE)
    with file_path.open("a") as d:
        d.write(f"{scene_name},{iteration},{psnr},{l1},{num_gaussians},{model_folder}\n")


def training(
        dataset,
        opt,
        pipe,
        rlp,
        wandb_config,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        run_name="",
        eval_output_path=None
):
    first_iter = 0
    # Get reward function to be used
    reward_function_name = rlp.reward_function
    print("Reward function used: ", reward_function_name)

    # Import the rewards module
    rewards_module = importlib.import_module("rewards.rewards")

    # Get the reward function from the module
    reward_function = getattr(rewards_module, reward_function_name)

    tb_writer = prepare_output_and_logger(dataset, wandb_config, eval_output_path)
    init_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, init_gaussians)

    wandb_logger.log_point_cloud(init_gaussians.initial_pcd.xyzrgb)
    
    init_gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        init_gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    gaussian_candidate_list = [init_gaussians]
    del init_gaussians
    gaussian_selection_rewards = [0]
    log_probability_candidates = None  # None for first iteration, as we have no log_probability_candidates
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Only use one candidate for evaluation optimization run
    k = rlp.num_candidates if rlp.train_rl else 1
    action_selector = ParamBasedActionSelector(k=k).to("cuda")
    policy_optimizer = torch.optim.AdamW(action_selector.parameters(), lr=rlp.rl_lr)
    lr_scheduler = StepLR(policy_optimizer, step_size=10, gamma=0.1)
    # Load RL meta model and optimizer
    if rlp.meta_model and Path(rlp.meta_model).exists():
        print(f"Loading meta_model from {rlp.meta_model}")
        action_selector.load_state_dict(torch.load(rlp.meta_model))

    if rlp.optimizer and Path(rlp.optimizer).exists():
        print(f"Loading optimizer from {rlp.optimizer}")
        policy_optimizer.load_state_dict(torch.load(rlp.optimizer))

    if rlp.lr_scheduler and Path(rlp.lr_scheduler).exists():
        print(f"Loading scheduler from {rlp.scheduler}")
        lr_scheduler.load_state_dict(torch.load(rlp.lr_scheduler))

    
    candidates_created = 0  # Counter when the last candidates were created
    for iteration in range(first_iter, opt.iterations + 1):
        if iteration - candidates_created > opt.densification_interval * 5:
            # Select best gaussians if densification is over
            gaussians_best_idx = torch.stack(gaussian_selection_rewards).argmax()
            #print(f"Rewards: {gaussian_selection_rewards}, idx: {gaussians_best_idx}")
            gaussians = gaussian_candidate_list[gaussians_best_idx]
            scene.gaussians = gaussians
            gaussian_candidate_list.clear()
            gaussian_selection_rewards.clear()
            gaussian_candidate_list.append(gaussians)
            gaussian_selection_rewards.append(0)
            candidates_created = opt.iterations + 1  # Do not do this again

        # Actual training start
        iter_start.record()

        for gaussians in gaussian_candidate_list:
            gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            for gaussians in gaussian_candidate_list:
                gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        for i, gaussians in enumerate(gaussian_candidate_list):
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
    
            loss.backward()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter]
            )

            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            # TODO: Calculate better reward for gaussian selection
            psnr_value = psnr(image, gt_image)
            reward = reward_function(loss, psnr_value, gaussians)
            gaussian_selection_rewards[i] = reward

            #TODO Rene implement wandblogger
            with torch.no_grad():
                wandb_logger.log_train_iter_candidate(iteration, i, gaussians, Ll1, ssim_value, loss, reward, image, gt_image)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "n_candidates": len(gaussian_candidate_list),
                    "n_gaussians": gaussians.num_points
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            scene.gaussians = gaussian_candidate_list[torch.stack(gaussian_selection_rewards).argmax()]
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if log_probability_candidates is not None:
                        # Update meta policy
                        # Rene: Make this true variable a parameter to controll if meta model should be learned or just used.
                        if rlp.train_rl:
                            with torch.enable_grad():
                                policy_optimizer.zero_grad(set_to_none=True)
                                
                                rewards = torch.stack(gaussian_selection_rewards).squeeze() # [Kandidaten]                                
                                advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8) #[Kandidaten]                                
                                expanded_advantage = advantage.unsqueeze(1).expand_as(log_probability_candidates) # [Kandidaten, Number Gaussians]   
                                # Maybe add entropy loss for exploration
                                # entropy_loss = -torch.mean(torch.sum(log_probability_candidates * torch.exp(log_probability_candidates), dim=1))                             
                                loss = -torch.mean(log_probability_candidates * expanded_advantage)
                                loss.backward()
                                policy_optimizer.step()

                    # Select best gaussian
                    gaussians_best_idx = torch.stack(gaussian_selection_rewards).argmax()
                    #print(f"Rewards: {gaussian_selection_rewards}, idx: {gaussians_best_idx}")
                    gaussians = gaussian_candidate_list[gaussians_best_idx]
                    scene.gaussians = gaussians
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    # Sample actions and create new candidates
                    with torch.enable_grad():
                        action_candidates, log_probability_candidates = action_selector(
                            gaussians,
                            iteration=iteration,
                            scene_extent=scene.cameras_extent
                        )

                    gaussian_candidate_list.clear()
                    gaussian_selection_rewards.clear()
                    wandb_logger.log_point_cloud(gaussians.point_cloud, iteration)
                    for i, actions in enumerate(action_candidates):
                        gaussian_clone = deepcopy(gaussians)
                        visualize_grad_scaling(gaussian_clone, name=f"Iteration {iteration:05d}:{i}", scene=scene, actions=actions)
                        n_cloned, n_splitted, n_pruned, n_gaussians, n_noop = apply_actions(gaussian_clone, actions, 0.005, size_threshold, scene.cameras_extent)
                        wandb_logger.log_densification_step(iteration, i, n_cloned, n_splitted, n_pruned, n_gaussians, n_noop)
                        gaussian_candidate_list.append(gaussian_clone)
                        gaussian_selection_rewards.append(0)

                    candidates_created = iteration


                    # OLD
                    # gaussians.densify_and_prune(
                    #     opt.densify_grad_threshold,
                    #     0.005,
                    #     scene.cameras_extent,
                    #     size_threshold,
                    # )

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    for gaussians in gaussian_candidate_list:
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                for gaussians in gaussian_candidate_list:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    Path(scene.model_path) / f"chkpnt{iteration}.pth",
                )
            # Check the number of gaussians and stop if necessary
            if gaussians.num_points > 300000 or gaussians.num_points < 200:
                print(f"\nNumber of gaussians {gaussians.num_points} is outside the range. Optimizing action selector and stopping.")
                if rlp.train_rl:
                    with torch.enable_grad():
                        policy_optimizer.zero_grad(set_to_none=True)
                        # Creating negativ reward for RL agent
                        # Get the shape of log_probability_candidates
                        num_can = log_probability_candidates.shape[0]
                        # WAS WORKING WITH TRAINNG but not eval rewards = torch.tensor([-1.0, -1.0], dtype=torch.float32, device="cuda")  # torch.Size([2])
                        rewards = torch.tensor([-1.0] * num_can, dtype=torch.float32, device="cuda")  # Shape [num_candidates]
                        advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                        expanded_advantage = advantage.unsqueeze(1).expand_as(log_probability_candidates)
                        loss = -torch.mean(log_probability_candidates * expanded_advantage)
                        loss.backward()
                        policy_optimizer.step()
                    
                    # Saving Meta Model as run will be stopped
                    if rlp.meta_model and rlp.train_rl:
                        save_model_optimizer_scheduler(rlp.meta_model, rlp.optimizer, rlp.lr_scheduler, action_selector, policy_optimizer, lr_scheduler)
                    break

    # Saving Meta Model
    if rlp.meta_model and rlp.train_rl:
        save_model_optimizer_scheduler(rlp.meta_model, rlp.optimizer, rlp.lr_scheduler, action_selector, policy_optimizer, lr_scheduler)


# Function to save model and optimizer states
def save_model_optimizer_scheduler(model_path, optimizer_path, scheduler_path, model, optimizer, scheduler):
    print(f"Saving meta model to {model_path}")
    torch.save(model.state_dict(), model_path)
    print(f"Saving optimizer state to {optimizer_path}")
    torch.save(optimizer.state_dict(), optimizer_path)
    print(f"Saving scheduler state to {scheduler_path}")
    torch.save(scheduler.state_dict(), scheduler_path)

# Unused function to be used to update the action selector
def rl_update(policy_optimizer, log_probability_candidates, gaussian_selection_rewards):
    with torch.enable_grad():
        policy_optimizer.zero_grad(set_to_none=True)
        
        rewards = torch.tensor(gaussian_selection_rewards, dtype=torch.float32, device="cuda") if isinstance(gaussian_selection_rewards, list) else torch.stack(gaussian_selection_rewards).squeeze()
        advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        expanded_advantage = advantage.unsqueeze(1).expand_as(log_probability_candidates)
        
        # Optionally, add entropy loss for exploration
        # entropy_loss = -torch.mean(torch.sum(log_probability_candidates * torch.exp(log_probability_candidates), dim=1))
        loss = -torch.mean(log_probability_candidates * expanded_advantage)
        
        loss.backward()
        policy_optimizer.step()

def apply_actions(gaussians: GaussianModel, actions: torch.Tensor, min_opacity, max_screen_size, extent):
    noop_mask = actions == 0
    clone_mask = actions == 1
    split_mask = actions == 2
    prune_mask = actions == 3

    # Extend split mask to have the correct size after cloning
    n_cloned_points = torch.sum(clone_mask)
    #print("SplitMaskDevice:", split_mask.device)
    split_mask = torch.cat(
        [
            split_mask,
            torch.zeros(n_cloned_points, device="cuda", dtype=torch.bool),
        ]
    )

    # Extend prune mask to have the correct size after cloning and splitting
    N = 2
    n_splitted_points = torch.sum(split_mask) * (N - 1)
    prune_mask = torch.cat(
        [
            prune_mask,
            torch.zeros(n_cloned_points + n_splitted_points, device="cuda", dtype=torch.bool),
        ]
    )
    n_pruned_points = torch.sum(prune_mask)
    n_noop_points = torch.sum(noop_mask)

    # Number of point before densification is done for correct logging
    n_gaussians = gaussians.num_points

    # Clone and split
    gaussians.densify_and_clone_selected(clone_mask)
    gaussians.densify_and_split_selected(split_mask, N=N)

    #print("PRUNE MASK ME: ", prune_mask.shape)
    gaussians.select_and_prune_points(prune_mask)
    #gaussians.select_and_prune_points_old(min_opacity, max_screen_size, extent)
    # Open the CSV file for appending
    #with open("densifcation.csv", mode='a', newline='') as log_file:
    #    writer = csv.writer(log_file)
    #    writer.writerow([0, n_cloned_points, n_splitted_points, torch.sum(prune_mask), gaussians.num_points])
    
    print(f"Cloned: {n_cloned_points}",
          f"Splitted: {n_splitted_points}",
          f"Pruned: {n_pruned_points}",
          f"NOOP: {torch.sum(noop_mask)}",
          f"NUMP: {n_gaussians}")
    
    torch.cuda.empty_cache()
    return n_cloned_points, n_splitted_points, n_pruned_points, n_gaussians, n_noop_points


def prepare_output_and_logger(args, wandb_config, eval_output_path=None):
    if eval_output_path:
        unique_str = wandb_config.name
        args.model_path = os.path.join(eval_output_path, unique_str)
    else:
        if not args.model_path:
            if os.getenv("OAR_JOB_ID"):
                unique_str = os.getenv("OAR_JOB_ID")
            else:
                unique_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # unique_str = str(uuid.uuid4())[0:10]
            args.model_path = os.path.join("./output/", unique_str)

    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    wdbp = WandbParams(parser)
    rlp = RLParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--eval_output_path", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(f"Optimizing {args.model_path}")
    # action_selector = GradNormThresholdSelector(init_threshold=0.17333875596523285 / 1000)
    # torch.save(action_selector.state_dict(),args.meta_model)

    # Initialize system state (RNG)
    safe_state(silent=args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # Example usage
    wandb_config = wdbp.extract(args)
    wandb_logger = WandBLogger(wandb_config)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        rlp.extract(args),
        wandb_config,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.run_name,
        args.eval_output_path
    )

    # All done
    print("\nTraining complete.")