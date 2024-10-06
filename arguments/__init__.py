#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Sequence, Union, Dict
from pathlib import Path

class GroupParams:
    def asdict(self):
        return {key: value for key, value in vars(self).items() if not key.startswith("_")}

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, nargs="+", type=str, default=value)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group
    
    def asdict(self):
        return {key: value for key, value in vars(self).items() if not key.startswith("_")}

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.start_number_gaussians = 100000
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

class RLParams(ParamGroup):
    def __init__(self, parser):
        self.rl_lr = 0.005
        self.meta_model = "meta_model.torch"
        self.base_model = "imitation_model.torch"
        self.optimizer = "rl_optimizer.torch"
        self.lr_scheduler = "lr_scheduler.torch"
        self.train_rl = False
        self.num_candidates = 2
        self.reward_function = ["reward_default"]
        self.hidden_size = 16
        self.break_reward = -10.0
        self.complexity_penalty = 0.001
        self.psnr_weight = 1.0
        self.late_reward_bonus = 0.00001
        self.increase_bias = 2.0
        self.decrease_bias = -2.0
        self.clip_param = 0.2
        self.ppo_update_frequency = 3

        super().__init__(parser, "RL Parameters")

class WandbParams(ParamGroup):
    def __init__(self, parser):
        self.project = "my-project"
        self.entity = ""
        self.name = ""
        self.id = ""
        self.job_type = None
        self.dir = None
        self.reinit = False
        self.tags = []
        self.group = ""
        self.notes = ""
        self.anonymous = ""
        self.mode = "offline"
        self.resume = "never"
        self.force = False
        self.save_code = True
        self.sync_tensorboard = False
        self.config = None

        super().__init__(parser, "WandB Init Config")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
