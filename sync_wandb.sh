#!/bin/bash

# Iterate over each directory in the current directory
for dir in */ ; do
    # Exclude the .submitit directory
    if [[ "$dir" != ".submitit/" ]]; then
        # Enter the directory
        cd "$dir"
        
        # Run the command with the desired parameters
        wandb sync --sync-all --append
        
        # Return to the parent directory
        cd ..
    fi
done