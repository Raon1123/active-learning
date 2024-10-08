import torch
import torch.optim as optim

from model.mlp import MLP

model_dict = {
    'mlp': MLP,
}

optim_dict = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
}

def get_model(config):
    model_name = config['MODEL']['model']
    
    assert model_name in model_dict, f"Model {model_name} not found"
    
    model_config = config['MODEL']['model_config']
    model = model_dict[model_name](**model_config)
    
    return model


def get_optimizer(config, model):
    optimizer_name = config['OPTIMIZER']['optimizer']
    
    assert optimizer_name in optim_dict, f"Optimizer {optimizer_name} not found"
    
    optimizer_config = config['OPTIMIZER']['optimizer_config']
    optimizer = optim_dict[optimizer_name](model.parameters(), **optimizer_config)
    
    return optimizer