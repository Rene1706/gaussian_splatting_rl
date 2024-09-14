import os
import sys
import hydra
from omegaconf import ListConfig, DictConfig
from hydra.utils import to_absolute_path
import subprocess
import random
import datetime
import pprint
from pathlib import Path
import random

# Get the current script directory
script_dir = Path(__file__).parent

def create_training_command(cfg) -> str:
    # Create the base command
    command = f"python {script_dir}/{cfg.training_script}"

    # Function to add parameters to the command
    def add_params(params, prefix=""):
        nonlocal command
        for key, value in params.items():
            if isinstance(value, bool):
                if value:
                    command += f" {prefix}--{key}"
            elif isinstance(value, (list, ListConfig)):
                # Handle lists correctly
                if len(value) > 0:
                    command += f" {prefix}--{key} {' '.join(map(str, value))}"
            elif value is not None:
                command += f" {prefix}--{key} {value}"

    # Add model_params, pipeline_params, and optimization_params to the command
    if cfg.rl_params.train_rl:
        add_params(cfg.optimization_params)
        add_params(cfg.pipeline_params)
        add_params(cfg.script_params)
    add_params(cfg.model_params)
    add_params(cfg.wandb_params)
    add_params(cfg.rl_params)

    return command

def get_datasets(data_path):
    datasets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    return datasets


def run_command(command, env=None, timeout=7200):  # Timeout set to 2 hours (7200 seconds)
    print(f"Executing: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Could also do this
    #process = subprocess.Popen(command, shell=True, stdout=stdout_file, stderr=stderr_file)
    #process.wait()  # Wait for the process to finish
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        print(f"Command timed out: {command}", file=sys.stderr)
        raise RuntimeError(f"Command timed out after {timeout} seconds")

    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    if process.returncode != 0:
        print("Error output:")
        print(stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with return code {process.returncode}: {command}")
"""
def run_command(command):
    with open('output.log', 'w') as stdout_file, open('error.log', 'w') as stderr_file:
        process = subprocess.Popen(command, shell=True, stdout=stdout_file, stderr=stderr_file)
        process.wait()  # Wait for the process to finish

        if process.returncode != 0:
            raise RuntimeError(f"Command failed with return code {process.returncode}")
"""

def train_and_evaluate(cfg, datasets, output_path):
    # Create log directory for this full evaluation run
    unique_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + str(random.randint(1000, 9999))
    full_eval_output_path = os.path.join("./output/", f"full_eval_{unique_str}")
    os.makedirs(full_eval_output_path, exist_ok=True)
    # Convert the relative path to an absolute path
    full_eval_output_path = os.path.abspath(full_eval_output_path)
    cfg.script_params.eval_output_path= full_eval_output_path
    print("Created output directory:", full_eval_output_path)

    full_data_path = os.path.abspath(full_eval_output_path)

    for epoch in range(cfg.eval_params.epochs):
        print(f"Running epoch {epoch + 1}/{cfg.eval_params.epochs}")
        # Sample random datasets for training and evaluation
        train_dataset = random.choice(datasets)
        eval_dataset = random.choice(datasets)
        print(f"Training on dataset {train_dataset}, evaluating on dataset {eval_dataset}")
        # RL Training
        if not cfg.eval_params.skip_training:
            #cfg.model_params.source_path = os.path.join(script_dir, cfg.eval_params.data_path, train_dataset)
            cfg.model_params.source_path = os.path.join(script_dir, cfg.eval_params.data_path, "01Gorilla")
            cfg.wandb_params.name = f"RL_train_{unique_str}"
            cfg.wandb_params.id = f"RL_train_{unique_str}"
            cfg.wandb_params.group = "default_pruning"
            cfg.wandb_params.tags = ["training", "default_pruning", f"reward_{cfg.rl_params.reward_function}", "no_buffer", f"lr{str(cfg.rl_params.rl_lr).replace('.', '_')}"]
            # Optimizing the RL model
            cfg.rl_params.train_rl = True
            training_command = create_training_command(cfg)
            run_command(training_command)

        # Optimization with RL model without learning
        if not cfg.eval_params.skip_eval and epoch % cfg.eval_params.eval_frequency == 0:
            cfg.model_params.source_path = os.path.join(script_dir, cfg.eval_params.data_path, "01Gorilla")#eval_dataset)
            cfg.wandb_params.name = f"RL_eval_{unique_str}"
            cfg.wandb_params.resume = "never"
            cfg.wandb_params.id = f"RL_eval_{unique_str}"
            cfg.wandb_params.group = "default_pruning"
            add_eval_tags(cfg)            
            # Skip optimizing the RL model
            cfg.rl_params.train_rl = False
            training_command = create_training_command(cfg)
            run_command(training_command)

def add_eval_tags(cfg):
    # Add tags for evaluation
    cfg.wandb_params.tags = ["evaluation", "default_pruning", "no_buffer", f"lr{str(cfg.rl_params.rl_lr).replace('.', '_')}"]
    if cfg.rl_params.meta_model:
        path = os.path.dirname(cfg.rl_params.meta_model)
        overrides_folder = os.path.join(path, ".hydra")
        if os.path.exists(overrides_folder):
            with open(os.path.join(overrides_folder, "overrides.yaml")) as f:
                model_params = f.read().splitlines()
                for line in model_params:
                    tag = line.split("=")[0].split(".")[1]
                    value = line.split("=")[1]
                    cfg.wandb_params.tags.append(f"{tag}_{value}")
                    if tag == "reward_function":
                        print("Setting reward function from override")
                        cfg.rl_params.reward_function = [value.strip('[]')]

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    data_path = to_absolute_path(cfg.eval_params.data_path)
    datasets = get_datasets(data_path)
    pprint.pprint(datasets)
    train_and_evaluate(cfg, datasets, "")

if __name__ == "__main__":
    main()


