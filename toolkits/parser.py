import os
import time

def exp_str(config, stamping=False):
    project_name = config['LOGGING']['project']
    postfix = config['LOGGING']['postfix']

    exp_list = [project_name, postfix]
    
    if stamping:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        exp_list.append(stamp)
    
    exp_str = "_".join(exp_list)
    
    return exp_str


def get_logdir(config, stamping=False):
    log_dir = dir_check(config['LOGGING']['log_dir'])
    
    exp = exp_str(config, stamping)
    
    log_dir = dir_check(os.path.join(log_dir, exp))
    
    return log_dir


def dir_check(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return directory