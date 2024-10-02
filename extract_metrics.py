import os
import json
import re
import argparse
def find_numbered_folder(output_folder):
    """Finds the folder inside 'output' that starts with a number."""
    for folder_name in os.listdir(output_folder):
        folder_path = os.path.join(output_folder, folder_name)
        if os.path.isdir(folder_path) and re.match(r'^\d+', folder_name):
            return folder_path
    return None

def find_run_id(wandb_folder):
    """Find the run ID from the folder name inside 'wandb'."""
    for folder_name in os.listdir(wandb_folder):
        if folder_name.startswith("offline-run-"):
            match = re.search(r'RL_eval_\d+_\d+_\d+_\d+', folder_name)
            if match:
                return match.group(0)
    return None

def extract_results_and_points(numbered_folder):
    """Extracts SSIM, PSNR from results.json and number of points from gaussian_num_points.txt."""
    results_file = os.path.join(numbered_folder, "results.json")
    points_file = os.path.join(numbered_folder, "gaussian_num_points.txt")

    # Initialize variables
    ssim = psnr = num_points = None

    # Read results.json
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
            ours_key = list(data.keys())[0]  # Assuming there's only one key like "ours_30000"
            ssim = data[ours_key].get("SSIM")
            psnr = data[ours_key].get("PSNR")

    # Read gaussian_num_points.txt
    if os.path.exists(points_file):
        with open(points_file, 'r') as f:
            line = f.readline().strip()
            match = re.search(r'\d+', line)
            if match:
                num_points = int(match.group(0))

    return ssim, psnr, num_points

def process_folders(base_folder):
    """Processes each folder, extracts run_id, SSIM, PSNR, and num_points."""
    results = []

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)

        # Skip if it's not a folder or if it's the .submit folder
        if not os.path.isdir(folder_path) or folder_name == '.submit':
            continue

        output_folder = os.path.join(folder_path, 'output')
        wandb_folder = os.path.join(folder_path, 'wandb')

        if not os.path.isdir(output_folder) or not os.path.isdir(wandb_folder):
            print(f"Required output or wandb folder not found in {folder_name}")
            continue

        # Find the numbered folder
        numbered_folder = find_numbered_folder(output_folder)
        if numbered_folder is None:
            print(f"No numbered folder found in {output_folder}")
            continue

        # Extract run_id from wandb folder
        run_id = find_run_id(wandb_folder)
        if run_id is None:
            print(f"Run ID not found in {wandb_folder}")
            continue

        # Extract SSIM, PSNR, and num_points
        ssim, psnr, num_points = extract_results_and_points(numbered_folder)

        if ssim is None or psnr is None or num_points is None:
            print(f"Missing data in {numbered_folder}")
            continue

        # Store results
        results.append({
            "run_id": run_id,
            "SSIM": ssim,
            "PSNR": psnr,
            "num_points": num_points
        })

    return results

def save_results_to_csv(results, output_file):
    """Saves results to a CSV file."""
    import csv
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['run_id', 'SSIM', 'PSNR', 'num_points']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process folders and extract metrics.")
    parser.add_argument('--base_folder', required=True, help='Path to the base folder')
    parser.add_argument('--output_file', required=True, help='Output file where results will be saved')
    args = parser.parse_args()

    base_folder = args.base_folder
    output_file = args.output_file

    # Process folders and extract data
    results = process_folders(base_folder)

    # Save results to CSV file
    if results:
        save_results_to_csv(results, output_file)
        print(f"Results saved to {output_file}")
    else:
        print("No results found.")
