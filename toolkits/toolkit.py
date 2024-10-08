import os

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