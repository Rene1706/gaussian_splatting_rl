import os
import shutil

def delete_files_media_and_events(base_folder):
    """
    This function traverses through the base folder, enters the 'wandb' subfolder, 
    deletes the 'files/media' folder within each subfolder of 'wandb', and deletes 
    any files that begin with 'events.out.tfevents' in 'output/<any_subfolder>/<sub_subfolder>'.

    Parameters:
    base_folder (str): The base folder where the script starts its search.
    """
    # Traverse the base folder
    for root, dirs, files in os.walk(base_folder):
        # Check if we are inside the first subfolder containing 'output'
        if 'output' in dirs:
            output_folder = os.path.join(root, 'output')
            # Go into the output folder and check subfolders
            for subfolder in os.listdir(output_folder):
                subfolder_path = os.path.join(output_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    # Go one level deeper inside the output subfolder
                    for sub_subfolder in os.listdir(subfolder_path):
                        sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
                        if os.path.isdir(sub_subfolder_path):
                            # Check for files starting with 'events.out.tfevents' and delete them
                            for file_name in os.listdir(sub_subfolder_path):
                                if file_name.startswith("events.out.tfevents"):
                                    file_path = os.path.join(sub_subfolder_path, file_name)
                                    try:
                                        os.remove(file_path)
                                        print(f"Deleted: {file_path}")
                                    except Exception as e:
                                        print(f"Error deleting {file_path}: {e}")

        # Check if we are inside a 'wandb' folder
        if os.path.basename(root) == 'wandb':
            # For every subfolder inside the 'wandb' folder
            for subfolder in dirs:
                media_folder = os.path.join(root, subfolder, 'files', 'media')
                
                # If the 'files/media' folder exists, delete it
                if os.path.exists(media_folder):
                    try:
                        shutil.rmtree(media_folder)
                        print(f"Deleted: {media_folder}")
                    except Exception as e:
                        print(f"Error deleting {media_folder}: {e}")

if __name__ == "__main__":
    # Set the base folder where the search begins
    base_folder = "/bigwork/nhmlhuer/git/backup/gs_rl/hydra/multirun/2024-10-16/19-47-10"  # Change to your specific base folder path

    # Call the function to delete the 'files/media' folder and event files in the first subfolder
    delete_files_media_and_events(base_folder)
