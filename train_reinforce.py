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
from random import randint, sample

import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, WandbParams, RLParams


from gaussian_renderer import network_gui, render
from scene import Scene, GaussianModel
from policies.action_selector import ParamBasedActionSelector
from replay_buffer import ReplayBuffer
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.reward_utils import exponential_moving_average
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
    # Initialize a buffer for storing (log_probs, reward) pairs
    max_buffer_size = 100000  # Buffer can hold up to 1,000,000 log_probs
    num_views = 50
    # Initialize buffers for storing log_probs and rewards
    replay_buffer = ReplayBuffer(max_buffer_size)
    # Load the replay buffer if it exists
    replay_buffer.load("replay_buffer.pth")
    
    densification_counter = 0

    # Initilize run
    first_iter = 0
    last_iter = 0
    last_iter_psnr = 0
    delta_gaussians = 0
    # Get reward function to be used
    reward_function_name = rlp.reward_function
    print("Reward function used: ", reward_function_name)

    # Import the rewards module
    rewards_module = importlib.import_module("rewards.rewards")

    # Construct the list of reward functions from the names in rlp.reward_function
    reward_functions = []
    for reward_name in rlp.reward_function:
        try:
            reward_func = getattr(rewards_module, reward_name)
            reward_functions.append(reward_func)
        except AttributeError:
            print(f"Warning: Reward function '{reward_name}' not found in the rewards module.")
            continue

    if not reward_functions:
        raise ValueError("No valid reward functions found. Please check the reward_function names in rl_params.")
    # Get the reward function from the module
    reward_function = reward_functions[0]  # Use the first one as the primary

    tb_writer = prepare_output_and_logger(dataset, wandb_config, eval_output_path)
    init_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, init_gaussians)
    last_iter = get_last_iteration(rlp.train_rl)
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
    inputs_candidates = None   # None for first iteration, as we have no inputs_candidates
    action_candidates = None            # None for first iteration, as we have no action_candidates
    gaussians_delta = [0]
    gaussian_selection_psnr = [0]
    log_probability_candidates = None  # None for first iteration, as we have no log_probability_candidates
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Only use one candidate for evaluation optimization run
    k = rlp.num_candidates if rlp.train_rl else 1

    # Initilize RL agent objects
    action_selector = ParamBasedActionSelector(k=k, hidden_size=rlp.hidden_size).to("cuda")
    policy_optimizer = torch.optim.AdamW(action_selector.parameters(), lr=rlp.rl_lr)
    #lr_scheduler = StepLR(policy_optimizer, step_size=10, gamma=0.1)
    #lr_scheduler = ExponentialLR(policy_optimizer, gamma=0.98)


    # Load RL meta model, optimizer and scheduler
    if rlp.meta_model and Path(rlp.meta_model).exists():
        print(f"Loading meta_model from {rlp.meta_model}")
        action_selector.load_state_dict(torch.load(rlp.meta_model))

    if rlp.optimizer and Path(rlp.optimizer).exists():
        print(f"Loading optimizer from {rlp.optimizer}")
        policy_optimizer.load_state_dict(torch.load(rlp.optimizer))

    #if rlp.lr_scheduler and Path(rlp.lr_scheduler).exists():
    #    print(f"Loading scheduler from {rlp.lr_scheduler}")
    #    lr_scheduler.load_state_dict(torch.load(rlp.lr_scheduler))

    
    candidates_created = 0  # Counter when the last candidates were created
    for iteration in range(first_iter, opt.iterations + 1):
        if iteration - candidates_created > opt.densification_interval * 5:
            # Check if there are valid rewards
            if gaussian_selection_rewards and len(gaussian_selection_rewards) > 0:
                # Convert rewards to tensors if they aren't already
                rewards_tensor = torch.stack([torch.tensor(r, dtype=torch.float32, device="cuda") for r in gaussian_selection_rewards])
                gaussians_best_idx = rewards_tensor.argmax()
                gaussians = gaussian_candidate_list[gaussians_best_idx]
                last_iter_psnr = gaussian_selection_psnr[gaussians_best_idx]
                scene.gaussians = gaussians
                gaussian_candidate_list.clear()
                gaussian_selection_rewards.clear()
                gaussian_selection_psnr.clear()
                gaussian_candidate_list.append(gaussians)
                gaussian_selection_rewards.append(0)
                gaussian_selection_psnr.append(0)
                candidates_created = iteration  # Update the timestamp
            else:
                # No valid rewards; skip or handle accordingly
                pass

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

            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # ! Render multiple images for RL agent reward
                    psnr_values = []
                    visibility_counts = torch.zeros(gaussians.num_points, device="cuda")

                    with torch.no_grad():
                        for _ in range(num_views):
                            # Pick a random camera for evaluation
                            eval_viewpoint_cam = scene.getTrainCameras()[randint(0, len(scene.getTrainCameras()) - 1)]

                            # Render without computing gradients
                            eval_render_pkg = render(eval_viewpoint_cam, gaussians, pipe, background)
                            eval_image, eval_visibility_filter = (
                                eval_render_pkg["render"],
                                eval_render_pkg["visibility_filter"],
                            )

                            # Compute PSNR
                            eval_gt_image = eval_viewpoint_cam.original_image.cuda()
                            eval_psnr = psnr(eval_image, eval_gt_image)
                            psnr_values.append(eval_psnr.mean().item())

                            # Accumulate visibility counts
                            visibility_counts[eval_visibility_filter] += 1.0

                    # Average PSNR over all views
                    average_psnr = sum(psnr_values) / num_views

                    # Normalize visibility counts
                    unique_counts = torch.unique(visibility_counts, return_counts=True)
                    print("Unique visibility counts: ", unique_counts)
                    visibility_counts = visibility_counts / num_views
                    
                    print("Shape: ", visibility_counts.shape)

                    reward = reward_function(loss=loss,
                                            psnr=average_psnr,
                                            last_psnr=last_iter_psnr,
                                            delta_gaussians=gaussians_delta[i],
                                            gaussians=gaussians,
                                            iteration=iteration,
                                            rl_params=rlp)
                    
                    # Update gaussian_selection_psnr[i] with the exponential moving average
                    gaussian_selection_rewards[i] = reward
                    gaussian_selection_psnr[i] = exponential_moving_average(gaussian_selection_psnr[i], average_psnr)
                    
                    # Compute per-Gaussian rewards
                    per_gaussian_rewards = torch.zeros(gaussians.num_points, device="cuda")
                    per_gaussian_rewards += reward * visibility_counts  # Weighted by visibility

                    # Map rewards to parent Gaussians
                    parent_indices = gaussians.parent_indices.cpu().numpy()
                    per_gaussian_rewards_np = per_gaussian_rewards.detach().cpu().numpy()

                    parent_rewards = {}
                    parent_counts = {}

                    for idx, parent_idx in enumerate(parent_indices):
                        if parent_idx not in parent_rewards:
                            parent_rewards[parent_idx] = 0.0
                            parent_counts[parent_idx] = 0
                        parent_rewards[parent_idx] += per_gaussian_rewards_np[idx]
                        parent_counts[parent_idx] += 1

                    # Average rewards for each parent Gaussian
                    for parent_idx in parent_rewards:
                        parent_rewards[parent_idx] /= parent_counts[parent_idx]
                    

                    # * Only log final reward before next densification
                    with torch.no_grad():
                        additional_rewards = {}
                        wandb_logger.log_train_iter_candidate(iteration, i, gaussians, Ll1, average_psnr, ssim_value, loss, reward, image, gt_image, additional_rewards)
                    # Update scene.gaussians with the best candidate
                    if gaussian_selection_rewards and len(gaussian_selection_rewards) > 0:
                        rewards_tensor = torch.stack([torch.tensor(r, dtype=torch.float32, device="cuda") for r in gaussian_selection_rewards])
                        best_idx = rewards_tensor.argmax()
                        scene.gaussians = gaussian_candidate_list[best_idx]
                    else:
                        scene.gaussians = gaussian_candidate_list[0]
            else:
                # No densification, maintain current gaussians
                scene.gaussians = gaussian_candidate_list[0]
        
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
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                if scene.gaussians is not None:
                    scene.save(iteration)
                else:
                    print("Warning: scene.gaussians is None, skipping save.")

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if action_candidates is not None:
                        densification_counter += 1
                        # Check each candidate and adjust reward if necessary
                        if rlp.train_rl:
                            for i, gaussians in enumerate(gaussian_candidate_list):
                                if gaussians.num_points > 300000 or gaussians.num_points < 100:
                                    gaussian_selection_rewards[i] = rlp.break_reward  # Set reward to -1 for this candidate

                        # Iterate through each candidate and store individual entries
                        for candidate_idx in range(len(action_candidates)):
                            actions = action_candidates[candidate_idx]  # Shape [100000]
                            old_log_probs = log_probability_candidates[candidate_idx]  # Shape [100000]
                            inputs = inputs_candidates  # Shape [100000, 3]
                            gaussians = gaussian_candidate_list[candidate_idx]
                            parent_indices = gaussians.parent_indices.cpu().numpy()

                            # Prepare data for replay buffer
                            inputs_np = inputs.cpu().numpy()
                            actions_np = actions.cpu().numpy()
                            log_probs_np = old_log_probs.cpu().numpy()

                            print("Length of inputs_np: ", len(inputs_np))
                            print("Length of actions_np: ", len(actions_np))
                            print("Length of log_probs_np: ", len(log_probs_np))

                            # Store per-Gaussian entries
                            for idx in range(len(parent_indices)):
                                parent_idx = parent_indices[idx]
                                input_tensor = torch.from_numpy(inputs_np[parent_idx]).to("cuda")
                                action= actions_np[parent_idx]
                                log_prob = log_probs_np[parent_idx]
                                reward = parent_rewards.get(parent_idx, 0.0)  # Reward for parent Gaussian
                                #print("Added : ", input, action, log_prob, reward)
                                replay_buffer.add(input_tensor, action, log_prob, reward)

                        break_training = False
                        if rlp.train_rl:
                            break_training = any(gaussian.num_points > 300000 or gaussian.num_points < 100 for gaussian in gaussian_candidate_list)
                        # Update meta policy
                        with torch.enable_grad():
                            if (densification_counter) % 3 == 0 and rlp.train_rl or break_training:
                                # Sample from the replay buffer
                                sampled_inputs, sampled_actions, sampled_old_log_probs, sampled_rewards = replay_buffer.sample(batch_size=max(1, int(0.3 * replay_buffer.size())))

                                # PPO update
                                ppo_loss = ppo_update(action_selector, policy_optimizer, sampled_inputs, sampled_actions, sampled_rewards, sampled_old_log_probs)
                                print("PPO loss:", ppo_loss)
                                print("Buffer size: ", replay_buffer.size())
                                # Optionally log RL loss, if needed
                                wandb_logger.log_rl_loss(iteration, ppo_loss, 0, policy_optimizer)

                        if break_training:
                            print(f"\nNumber of gaussians is outside the range. Optimizing action selector and stopping.")
                            last_iter += iteration
                            # Saving Meta Model as run will be stopped (Not needed as with break it will get saved regardless)
                            # if rlp.meta_model and rlp.train_rl: 
                                #save_model_optimizer_scheduler(rlp.meta_model, rlp.optimizer, rlp.lr_scheduler, action_selector, policy_optimizer, lr_scheduler)
                            break

                    # Select best gaussian
                    gaussians_best_idx = torch.stack(gaussian_selection_rewards).argmax()
                    #print(f"Rewards: {gaussian_selection_rewards}, idx: {gaussians_best_idx}")
                    gaussians = gaussian_candidate_list[gaussians_best_idx]
                    # Reset parent_indices to match current indices
                    gaussians.parent_indices = torch.arange(gaussians.num_points, device="cuda")
                    last_iter_psnr = gaussian_selection_psnr[gaussians_best_idx]
                    scene.gaussians = gaussians
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    # Sample actions and create new candidates
                    with torch.enable_grad():
                        action_candidates, inputs_candidates, log_probability_candidates = action_selector(
                            gaussians,
                            iteration=iteration,
                            scene_extent=scene.cameras_extent
                        )

                    gaussian_candidate_list.clear()
                    gaussian_selection_rewards.clear()
                    gaussian_selection_psnr.clear()
                    gaussians_delta.clear()
                    if iteration % 5000 == 0 and not rlp.train_rl:
                        wandb_logger.log_point_cloud(gaussians.point_cloud, iteration)
                    for i, actions in enumerate(action_candidates):
                        gaussian_clone = deepcopy(gaussians)
                        #visualize_grad_scaling(gaussian_clone, name=f"Iteration {iteration:05d}:{i}", scene=scene, actions=actions)
                        n_cloned, n_splitted, n_pruned, n_gaussians, n_noop = apply_actions(gaussian_clone, actions, 0.005, size_threshold, scene.cameras_extent)
                        delta_gaussians = n_cloned + n_splitted - n_pruned
                        gaussians_delta.append(delta_gaussians)
                        wandb_logger.log_densification_step(iteration, i, n_cloned, n_splitted, n_pruned, n_gaussians, n_noop)
                        gaussian_candidate_list.append(gaussian_clone)
                        gaussian_selection_rewards.append(0)
                        gaussian_selection_psnr.append(0)

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
    last_iter += iteration

    # Saving Meta Model
    if rlp.meta_model and rlp.train_rl:
        lr_scheduler = None
        save_model_optimizer_scheduler(rlp.meta_model, rlp.optimizer, rlp.lr_scheduler, action_selector, policy_optimizer, lr_scheduler)
        replay_buffer.save("replay_buffer.pth")
    # Save iterations so logging can be done in one run
    save_last_iteration(rlp.train_rl, last_iter)


# Function to save model and optimizer states
def save_model_optimizer_scheduler(model_path, optimizer_path, scheduler_path, model, optimizer, scheduler):
    print(f"Saving meta model to {model_path}")
    torch.save(model.state_dict(), model_path)
    print(f"Saving optimizer state to {optimizer_path}")
    torch.save(optimizer.state_dict(), optimizer_path)
    #print(f"Saving scheduler state to {scheduler_path}")
    #torch.save(scheduler.state_dict(), scheduler_path)

def save_last_iteration(train_rl, iteration):
    print(f"Saving last iteration: {iteration}")
    if train_rl:
        iteration_file = Path("last_iteration_train.txt")
    else: 
        iteration_file = Path("last_iteration_eval.txt")
    with iteration_file.open("w") as f:
        f.write(str(iteration))

def get_last_iteration(train_rl):
    if train_rl:
        iteration_file = Path("last_iteration_train.txt")
    else: 
        iteration_file = Path("last_iteration_eval.txt")
    if iteration_file.exists():
        with iteration_file.open("r") as f:
            return int(f.read().strip())
    return 0

def apply_actions(gaussians: GaussianModel, actions: torch.Tensor, min_opacity, max_screen_size, extent):
    noop_mask = actions == 0
    clone_mask = actions == 1
    split_mask = actions == 2

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
    n_noop_points = torch.sum(noop_mask)

    # Number of point before densification is done for correct logging
    n_gaussians = gaussians.num_points

    # Clone and split
    gaussians.densify_and_clone_selected(clone_mask)
    gaussians.densify_and_split_selected(split_mask, N=N)

    #gaussians.select_and_prune_points(prune_mask)
    # ? Pruning done after split, clone as otherwhise masks are not accurate anymore.
    # ? Problem is that often cloned points are directly pruned afterwards
    n_pruned_points = gaussians.select_and_prune_points_old(min_opacity, max_screen_size, extent)

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

def create_wandb_config(args):
    # Convert the args Namespace to a dictionary
    args_dict = vars(args)
    # Filter out excluded keys
    wandb_config = {key: value for key, value in args_dict.items()}
    
    return wandb_config

def ppo_update(policy_net, optimizer, sampled_inputs, sampled_actions, sampled_rewards, sampled_log_probs, clip_param=0.2):
    # Get new log probabilities from the current policy
    new_policy = policy_net.get_policy(sampled_inputs)
    new_log_probs = new_policy.log_prob(sampled_actions)

    # Calculate the mean reward as the baseline
    baseline = sampled_rewards.mean()

    # Compute advantages using the mean reward as the baseline
    advantages = sampled_rewards - baseline  # Simplified advantage calculation

    # Normalize advantages for stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Compute the ratio of new and old probabilities
    ratio = torch.exp(new_log_probs - sampled_log_probs)

    # Clipped loss for PPO
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Total loss (PPO policy loss only)
    loss = policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

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

    # Initialize system state (RNG)
    safe_state(silent=args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # Example usage
    wandb_args = wdbp.extract(args)
    rlp_config = rlp.extract(args)
    lp_config = lp.extract(args)
    last_iteration = get_last_iteration(rlp_config.train_rl)
    wandb_config = create_wandb_config(args)
    wandb_logger = WandBLogger(wandb_args, wandb_config, last_iteration)
    training(
        lp_config,
        op.extract(args),
        pp.extract(args),
        rlp_config,
        wandb_args,
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