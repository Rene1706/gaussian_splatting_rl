import os
import subprocess
import re
import shutil

def find_numbered_folder(output_folder):
    """Finds the folder inside 'output' that starts with a number."""
    for folder_name in os.listdir(output_folder):
        folder_path = os.path.join(output_folder, folder_name)
        if os.path.isdir(folder_path) and re.match(r'^\d+', folder_name):
            return folder_path  # Return the path to the folder starting with a number
    return None

def delete_train_test_folders(numbered_folder):
    """Deletes 'train' and 'test' folders inside the numbered folder."""
    train_folder = os.path.join(numbered_folder, 'train')
    test_folder = os.path.join(numbered_folder, 'test')
    
    # Remove 'train' folder if it exists
    if os.path.exists(train_folder) and os.path.isdir(train_folder):
        shutil.rmtree(train_folder)
        print(f"Deleted {train_folder}")
    
    # Remove 'test' folder if it exists
    if os.path.exists(test_folder) and os.path.isdir(test_folder):
        shutil.rmtree(test_folder)
        print(f"Deleted {test_folder}")

def run_scripts_in_folders(base_folder):
    # Walk through the directories in the base folder
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        # Skip if it's not a folder or if it's the .submit folder
        if not os.path.isdir(folder_path) or folder_name == '.submit':
            continue
        
        # Look for 'output' folder inside each folder
        output_folder = os.path.join(folder_path, 'output')
        if not os.path.isdir(output_folder):
            print(f"Output folder not found in {folder_name}")
            continue
        
        # Find the folder that starts with a number
        numbered_folder = find_numbered_folder(output_folder)
        if numbered_folder is None:
            print(f"No numbered folder found in {output_folder}")
            continue

        # Run render.py with -m <numbered_folder>
        command1 = f'python render.py -m {numbered_folder}'
        try:
            result1 = subprocess.run(command1, shell=True, capture_output=True, text=True)
            print(f"Ran {command1} with output:\n{result1.stdout}")
            if result1.stderr:
                print(f"Errors from {command1}:\n{result1.stderr}")
        except Exception as e:
            print(f"Error while running render.py in {numbered_folder}: {str(e)}")

        # Run metrics.py with -m <numbered_folder>
        command2 = f'python metrics.py -m {numbered_folder}'
        try:
            result2 = subprocess.run(command2, shell=True, capture_output=True, text=True)
            print(f"Ran {command2} with output:\n{result2.stdout}")
            if result2.stderr:
                print(f"Errors from {command2}:\n{result2.stderr}")
        except Exception as e:
            print(f"Error while running metrics.py in {numbered_folder}: {str(e)}")

        # Delete 'train' and 'test' folders after both scripts have been run
        delete_train_test_folders(numbered_folder)

if __name__ == "__main__":
    base_folder = "hydra/multirun/eval_grad_not_norm_ppo/10-08-36"  # Replace with the path to your base folder
    run_scripts_in_folders(base_folder)
