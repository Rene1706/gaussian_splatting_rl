import os
from argparse import ArgumentParser
import sys
from hydra import initialize, compose
from omegaconf import OmegaConf, ListConfig
import subprocess
import random
import wandb
import json

def create_training_command(cfg) -> str:
    # Create the base command
    command = cfg.training_script

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

    return command

def get_datasets(data_path):
    datasets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    return datasets


def run_command(command, env=None):
    print(f"Executing: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    # Read the output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Check for errors
    stderr_output, _ = process.communicate()
    if process.returncode != 0:
        print("Error output:")
        print(stderr_output)
        raise RuntimeError(f"Command failed with return code {process.returncode}: {command}")

def train_and_evaluate(cfg, datasets, output_path):
    for iteration in range(cfg.eval_params.iterations):
        print(f"Running iteration {iteration + 1}/{cfg.eval_params.iterations}")
        
        # Sample random datasets for training and evaluation
        train_dataset = random.choice(datasets)
        eval_dataset = random.choice(datasets)

        # RL Training
        if not cfg.eval_params.skip_training:
            cfg.model_params.source_path = os.path.join(cfg.eval_params.data_path, train_dataset)
            cfg.script_params.group = "train"
            training_command = create_training_command(cfg)
            run_command(training_command, env=os.environ.copy())

        # Optimization with RL model without learning
        if not cfg.eval_params.skip_eval:
            cfg.model_params.source_path = os.path.join(cfg.eval_params.data_path, eval_dataset)
            cfg.script_params.group = "eval"
            training_command = create_training_command(cfg)
            run_command(training_command, env=os.environ.copy())

        # Evaluation
        #if not cfg.eval_params.skip_rendering or not cfg.eval_params.skip_metrics:
        #    cfg.model_params.source_path = os.path.join(cfg.eval_params.data_path, eval_dataset)
        #    cfg.model_params.model_path = os.path.join(output_path, f"eval_{eval_dataset}")

            #if not cfg.eval_params.skip_rendering:
                #rendering_command = f"python render.py --model_path {cfg.model_params._model_path}"
                #run_command(rendering_command)
            
            #if not cfg.eval_params.skip_metrics:
                #metrics_command = f"python metrics.py --model_path {cfg.model_params._model_path}"
                #run_command(metrics_command)

                # Log metrics (this assumes metrics.py produces a JSON file with metrics)
                #metrics_file = os.path.join(cfg.model_params._model_path, "metrics.json")
                #with open(metrics_file) as f:
                #    metrics = json.load(f)
                #    wandb.log(metrics)
                #wandb.finish()

if __name__ == "__main__":
    config_name = "config"  # Change this to your desired config name
    # Initialize Hydra and compose the configuration
    with initialize(config_path="conf"):
        cfg = compose(config_name=config_name)
    
    datasets = get_datasets(cfg.eval_params.data_path)
    print(datasets)
    train_and_evaluate(cfg, datasets, "")

