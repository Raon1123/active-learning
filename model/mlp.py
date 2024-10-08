from typing import (
    List,
    Optional,
)

import torch
from torch import nn

# Multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, 
                 dims: List[int],
                 dropout: Optional[float] = None,
                 activation: Optional[nn.Module] = nn.ReLU()):
        r"""
        Simple multi-layer perceptron baseline.
        
        Args:
            dims: List of dimensions for each layer.
            dropout: Dropout rate.
            activation: Activation function
            
        Example:
            >>> model = MLP([784, 128, 128, 10],
            >>>             dropout=0.2,
            >>>             activation=nn.ReLU())
        """
        super(MLP, self).__init__()
        
        module_list = []
        
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            module_list.append(nn.Linear(in_dim, out_dim))
            if activation is not None:
                module_list.append(activation)
            if dropout is not None:
                module_list.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(module_list) 
        
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)
    
    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)
    