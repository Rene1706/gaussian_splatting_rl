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
    """Extracts num_points from a gaussian_num_points_<datasetname>.json or .txt file."""
    num_points = None
    if points_file_path.endswith('.json'):
        with open(points_file_path, 'r') as f:
            data = json.load(f)
            num_points = data.get('num_points')
    else:
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
    return params

def process_folders(base_folder):
    """Recursively processes folders, extracts data, and returns a list of dictionaries."""
    results = []

    for root, dirs, files in os.walk(base_folder):
        result_files = [f for f in files if f.startswith('results_') and f.endswith('.json')]
        points_files = [f for f in files if f.startswith('gaussian_num_points_') and (f.endswith('.json') or f.endswith('.txt'))]

        for result_file in result_files:
            dataset_name = result_file[len('results_'):-len('.json')]

            possible_points_files = [pf for pf in points_files if pf.startswith(f'gaussian_num_points_{dataset_name}')]
            if possible_points_files:
                points_file = possible_points_files[0]
            else:
                print(f"No points file found for dataset {dataset_name} in {root}")
                continue

            result_file_path = os.path.join(root, result_file)
            points_file_path = os.path.join(root, points_file)

            ssim, psnr = extract_metrics_from_result_file(result_file_path)
            num_points = extract_num_points_from_points_file(points_file_path)

            hydra_folder = find_hydra_folder(root)
            if hydra_folder is None:
                print(f"No .hydra folder found for {root}")
                continue
            overrides_file = os.path.join(hydra_folder, 'overrides.yaml')
            if not os.path.exists(overrides_file):
                print(f"No overrides.yaml found in {hydra_folder}")
                continue
            other_params = extract_parameters_from_overrides(overrides_file)

            # Use the meta_model path to find another overrides.yaml
            if 'meta_model' in other_params:
                meta_model_path = other_params['meta_model']
                meta_model_path = meta_model_path.strip('"\'')
                # Remove the filename to get the directory
                meta_model_dir = os.path.dirname(meta_model_path)
                meta_hydra_folder = os.path.join(meta_model_dir, '.hydra')
                meta_overrides_file = os.path.join(meta_hydra_folder, 'overrides.yaml')
                if os.path.exists(meta_overrides_file):
                    print(meta_overrides_file)
                    meta_params = extract_parameters_from_overrides(meta_overrides_file)
                    print(meta_params)
                    # Prefix meta parameters to distinguish them
                    for key, value in meta_params.items():
                        other_params[f'meta_{key}'] = value
                else:
                    print(f"No overrides.yaml found in {meta_hydra_folder}")
            else:
                print(f"meta_model not found in overrides.yaml at {overrides_file}")

            data_entry = {
                'dataset_name': dataset_name,
                'SSIM': ssim,
                'PSNR': psnr,
                'num_points': num_points,
                **other_params
            }

            results.append(data_entry)

    return results

def save_results_to_csv(results, output_file):
    """Saves results to a CSV file."""
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_json(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process folders and extract metrics.")
    parser.add_argument('--base_folder', required=True, help='Path to the base folder')
    parser.add_argument('--output_file', required=True, help='Output file where results will be saved')
    args = parser.parse_args()

    base_folder = args.base_folder
    output_file = args.output_file

    results = process_folders(base_folder)

    if results:
        save_results_to_csv(results, output_file)
        print(f"Results saved to {output_file}")
    else:
        print("No results found.")
