from typing import (
    Callable,
    List,
    Optional,
    
)

import numpy as np
import torch
from torch.utils.data import Dataset


# Active learning dataset abstraction
class ActiveLearningDataset(Dataset):
    def __init__(self,
                 samples: List[torch.Tensor | np.ndarray],
                 targets: Optional[List[torch.Tensor | np.ndarray]] = None,
                 transform: Optional[Callable] = None,
                 dataloader_config: Optional[dict] = None):
        r"""Dataset for active learning.
        
        """
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).float()
        self.samples = samples
    
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        self.targets = targets
    
        self.transform = transform
        self.dataloader_config = dataloader_config
        
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
    
    def add_label(self,
                  idx: int | List[int],):
        if isinstance(idx, int):
            idx = [idx]
        self.labeled_indices.extend(idx)
    
    def get_unlabeled_index(self):
        return list(set(range(len(self.samples))) - set(self.labeled_indices))
    
    def get_dataloader(self):
        return torch.utils.data.DataLoader(self, **self.dataloader_config)