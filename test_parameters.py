from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
# Create a parser
parser = ArgumentParser(description="Example script using parameter groups")

# Initialize parameter groups
model_params = ModelParams(parser)
pipeline_params = PipelineParams(parser)
optimization_params = OptimizationParams(parser)

# Combine arguments from command-line and config file
#args = get_combined_args(parser)

# Extract and use the parameters
#model_params_group = model_params.extract(args)
#pipeline_params_group = pipeline_params.extract(args)
#optimization_params_group = optimization_params.extract(args)

# Now you can use the extracted parameters
print("Model Params:", vars(model_params))
print("Pipeline Params:", vars(pipeline_params))
print("Optimization Params:", vars(optimization_params))