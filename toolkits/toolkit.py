import os

def is_path(path):
    # reference: https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(f"{path} is not a directory")