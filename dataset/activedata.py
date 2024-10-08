from typing import (
    Callable,
    List,
    Optional,
)

import torch
from torch.utils.data import Dataset


# Active learning dataset abstraction
class ActiveLearningDataset(Dataset):
    def __init__(self,
                 samples: List[torch.Tensor],
                 targets: Optional[List[torch.Tensor]] = None,
                 transform: Optional[Callable] = None,):
        r"""Dataset for active learning.
        
        """
        self.samples = samples
        self.targets = targets
        self.transform = transform
        
        self.labeled_indices = []
    
    def __len__(self):
        return len(self.labeled_indices)
    
    def __getitem__(self, idx):
        idx = self.labeled_indices[idx]
        
        sample = self.samples[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target
    
    def _label_data(self, idx, label):
        self.targets[idx] = label
    
    def add_label(self, idx):
        self.labeled_indices.append(idx)
    
    def get_unlabeled_index(self):
        return list(set(range(len(self.samples))) - set(self.labeled_indices))[0]