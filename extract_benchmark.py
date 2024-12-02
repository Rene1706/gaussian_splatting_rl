import os
import json
import re
import argparse

def extract_metrics_from_result_file(result_file_path):
    """Extracts SSIM and PSNR from a results_<datasetname>.json file."""
    with open(result_file_path, 'r') as f:
        data = json.load(f)
        if data:
            first_key = list(data.keys())[0]
            ssim = data[first_key].get('SSIM')
            psnr = data[first_key].get('PSNR')
            return ssim, psnr
    return None, None

def extract_num_points_from_points_file(points_file_path):
    """Extracts num_points from a gaussian_num_points.txt file."""
    num_points = None
    with open(points_file_path, 'r') as f:
        line = f.readline().strip()
        match = re.search(r'\d+', line)
        if match:
            num_points = int(match.group(0))
    return num_points

def find_hydra_folder(current_path):
    """Finds the .hydra folder by traversing up the directory tree."""
    while True:
        hydra_folder = os.path.join(current_path, '.hydra')
        if os.path.isdir(hydra_folder):
            return hydra_folder
        parent = os.path.dirname(current_path)
        if parent == current_path:
            return None
        current_path = parent

def extract_parameters_from_overrides(overrides_file):
    """Extracts specified parameters from overrides.yaml."""
    params = {}
    with open(overrides_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.lstrip('- ').strip()
                value = value.strip().strip('"\'')  # Remove surrounding quotes
                if key.startswith('rl_params.'):
                    param_name = key[len('rl_params.'):]
                    params[param_name] = value
                elif key.startswith('model_params.source_path'):
                    # Extract dataset name from the source_path
                    dataset_path = value
                    dataset_name = os.path.basename(os.path.normpath(dataset_path))
                    params['dataset_name'] = dataset_name
                else:
                    params[key] = value
    return params

def get_point_cloud_size(root):
    """Gets the size of point_cloud.ply in MB."""
    point_cloud_path = os.path.join(root, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
    if os.path.isfile(point_cloud_path):
        size_bytes = os.path.getsize(point_cloud_path)
        size_mb = size_bytes / (1024 * 1024)  # Convert bytes to MB
        return size_mb
    else:
        print(f"Point cloud file not found at {point_cloud_path}")
        return None

def process_folders(base_folder):
    """Recursively processes folders, extracts data, and returns a list of dictionaries."""
    results = []

    for root, dirs, files in os.walk(base_folder):
        if 'results.json' in files and 'gaussian_num_points.txt' in files:
            result_file_path = os.path.join(root, 'results.json')
            points_file_path = os.path.join(root, 'gaussian_num_points.txt')

            ssim, psnr = extract_metrics_from_result_file(result_file_path)
            num_points = extract_num_points_from_points_file(points_file_path)

            # Get point cloud size
            point_cloud_size_mb = get_point_cloud_size(root)

            hydra_folder = find_hydra_folder(root)
            if hydra_folder is None:
                print(f"No .hydra folder found for {root}")
                continue
            overrides_file = os.path.join(hydra_folder, 'overrides.yaml')
            if not os.path.exists(overrides_file):
                print(f"No overrides.yaml found in {hydra_folder}")
                continue
            other_params = extract_parameters_from_overrides(overrides_file)

            # If 'dataset_name' was not found in overrides.yaml, set a default or skip
            dataset_name = other_params.get('dataset_name', 'unknown_dataset')

            data_entry = {
                'dataset_name': dataset_name,
                'SSIM': ssim,
                'PSNR': psnr,
                'num_points': num_points,
                'point_cloud_size_mb': point_cloud_size_mb,
                **other_params
            }

            results.append(data_entry)

    return results

def save_results_to_json(results, output_file):
    """Saves results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process folders and extract metrics.")
    parser.add_argument('--base_folder', required=True, help='Path to the base folder')
    parser.add_argument('--output_file', required=True, help='Output file where results will be saved')
    args = parser.parse_args()

    base_folder = args.base_folder
    output_file = args.output_file

    results = process_folders(base_folder)

    if results:
        save_results_to_json(results, output_file)
        print(f"Results saved to {output_file}")
    else:
        print("No results found.")
