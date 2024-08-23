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
    add_params(cfg.model_params)
    add_params(cfg.pipeline_params)
    add_params(cfg.optimization_params)
    add_params(cfg.script_params)
    add_params(cfg.wandb_params)
    add_params(cfg.rl_params)

    return command

def get_datasets(data_path):
    datasets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    return datasets


def run_command(command, env=None):
    print(f"Executing: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    stdout_lines = []
    stderr_lines = []

    while True:
        stdout_output = process.stdout.readline()
        stderr_output = process.stderr.readline()

        if stdout_output == '' and stderr_output == '' and process.poll() is not None:
            break

        if stdout_output:
            print(stdout_output.strip())
            stdout_lines.append(stdout_output.strip())

        if stderr_output:
            print(stderr_output.strip(), file=sys.stderr)
            stderr_lines.append(stderr_output.strip())

    # Check for errors
    process.wait()
    if process.returncode != 0:
        print("Error output:")
        for line in stderr_lines:
            print(line, file=sys.stderr)
        raise RuntimeError(f"Command failed with return code {process.returncode}: {command}")

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
            cfg.model_params.source_path = os.path.join(script_dir, cfg.eval_params.data_path, train_dataset)
            cfg.wandb_params.name = f"RL_train_{unique_str}"
            cfg.wandb_params.id = f"RL_train_{unique_str}"
            cfg.wandb_params.group = "default_pruning"
            cfg.wandb_params.tags = ["training", "default_pruning", f"reward_{cfg.rl_params.reward_function}", "buffer", f"lr{str(cfg.rl_params.rl_lr).replace('.', '_')}"]
            # Optimizing the RL model
            cfg.rl_params.train_rl = True
            training_command = create_training_command(cfg)
            run_command(training_command, env=os.environ.copy())

        # Optimization with RL model without learning
        if not cfg.eval_params.skip_eval and epoch % cfg.eval_params.eval_frequency == 0:
            cfg.model_params.source_path = os.path.join(script_dir, cfg.eval_params.data_path, eval_dataset)
            cfg.wandb_params.name = f"RL_eval_{unique_str}"
            cfg.wandb_params.id = f"RL_eval_{unique_str}"
            cfg.wandb_params.group = "default_pruning"
            cfg.wandb_params.tags = ["evaluation", "default_pruning", f"reward_{cfg.rl_params.reward_function}", "buffer", f"lr{str(cfg.rl_params.rl_lr).replace('.', '_')}"]
            # Skip optimizing the RL model
            cfg.rl_params.train_rl = False
            training_command = create_training_command(cfg)
            run_command(training_command, env=os.environ.copy())


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    data_path = to_absolute_path(cfg.eval_params.data_path)
    datasets = get_datasets(data_path)
    pprint.pprint(datasets)
    train_and_evaluate(cfg, datasets, "")

if __name__ == "__main__":
    main()


