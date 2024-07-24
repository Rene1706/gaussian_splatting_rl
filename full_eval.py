import os
from argparse import ArgumentParser

# Define all scenes
scenes = [
    "01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", 
    "07Owl", "08Sabertooth", "09Swan", "10Sheep", "11Pig", "12Zalika", 
    "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion", "18Obesobeso", 
    "19Bear", "20Puppy"
]

def parse_arguments():
    parser = ArgumentParser(description="Full evaluation script parameters")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_rendering", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--output_path", default="./eval")
    parser.add_argument('--data_path', required=True, type=str, help="Path to the dataset base directory")
    args = parser.parse_args()
    return args

def train_models(args):
    if args.skip_training:
        return

    common_args = " --quiet --eval --test_iterations -1 "
    
    for scene in scenes:
        source = os.path.join(args.data_path, scene)
        output = os.path.join(args.output_path, scene)
        os.system(f"python train.py -s {source} -m {output} {common_args}")
