import torch.nn as nn
import torch.optim as optim
from torchvision import models

from model.mlp import MLP

PRETRAIN_TAG = ['resnet']

model_dict = {
    'mlp': MLP,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
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
    
    if any([tag in model_name for tag in PRETRAIN_TAG]):
        # change the last layer to fit the number of classes
        num_classes = config['DATASET']['num_classes']
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    return model


def get_optimizer(config, model):
    optimizer_name = config['MODEL']['optimizer']
    
    assert optimizer_name in optim_dict, f"Optimizer {optimizer_name} not found"
    
    optimizer_config = config['MODEL']['optimizer_config']
    optimizer = optim_dict[optimizer_name](model.parameters(), **optimizer_config)
    
    return optimizer