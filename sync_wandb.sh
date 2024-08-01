#!/bin/bash

# Set the root directory containing your training run output folders
ROOT_DIR="."

# Find all 'wandb' folders and sync them
find "$ROOT_DIR" -type d -name 'wandb' | while read -r wandb_folder; do
    echo "Syncing $wandb_folder..."
    wandb sync "$wandb_folder"
    if [ $? -eq 0 ]; then
        echo "Successfully synced $wandb_folder"
    else
        echo "Failed to sync $wandb_folder"
    fi
done
