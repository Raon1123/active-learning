import os

import numpy as np
import torch
import yaml

def is_path(path):
    # reference: https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
    if os.path.exists(path):
        return path
    else:
        raise NotADirectoryError(f"{path} is not a directory")
    

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    return config


def seed_fixing(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)