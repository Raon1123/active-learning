from typing import (
    List,
    Optional,
)

import numpy as np

class RandomSampler():
    def __init__(self):
        pass
    
    def sample(self, 
               dataset, 
               n_samples: int) -> List[int]:
        r"""Randomly sample data from the dataset.
        """
        unlabel_data = dataset.get_unlabeled_index()
        choices = np.random.choice(unlabel_data, n_samples, replace=False)
        
        return choices