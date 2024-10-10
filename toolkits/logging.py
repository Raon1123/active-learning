import os

from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
except ImportError:
    wandb = None

from toolkits.parser import (
    get_logdir
)

writer_dir = {
    'tensorboard': SummaryWriter,
    'wandb': wandb
}

def init_logger(config):
    log_dir = get_logdir(config, stamping=True)
    logger = config['LOGGING']['logger']
    
    assert logger in writer_dir, f"Logger {logger} not found"
    
    if logger == 'tensorboard':
        writer = writer_dir[logger](log_dir)
    elif wandb is not None and logger == 'wandb':
        writer = writer_dir[logger].init(project=config['LOGGING']['project'],
                                         name=config['LOGGING']['postfix'],
                                         dir=log_dir,
                                         config=config)
        
    return writer, log_dir
    

def log_scalar(writer, tag, value, step):
    writer.add_scalar(tag, value, step)
    
    
def log_index(value, step, log_dir):
    file_path = os.path.join(log_dir, 'index_{}.log'.format(step))
    
    with open(file_path, 'w') as f:
        f.write(str(value))