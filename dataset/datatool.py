import torch
import torchvision.datasets as tvdatasets

from dataset.activedata import ActiveLearningDataset

data_dict = {
    'mnist': tvdatasets.MNIST,
    'cifar10': tvdatasets.CIFAR10,
    'cifar100': tvdatasets.CIFAR100,
}

def get_dataset(config):
    dataset_name = config['DATASET']['dataset']
    
    assert dataset_name in data_dict, f"Dataset {dataset_name} not found"
    
    loader_config = config['DATASET']['dataloader']
    
    train_dataset = data_dict[dataset_name](root=config['DATASET']['data_root'], 
                                            train=True, download=True)
    test_dataset = data_dict[dataset_name](root=config['DATASET']['data_root'],
                                             train=False, download=True)
    
    loader_config['shuffle'] = True
    train_active_dataset = ActiveLearningDataset(samples=train_dataset.data,
                                                 targets=train_dataset.targets,
                                                 dataloader_config=loader_config)
    
    loader_config['shuffle'] = False
    test_active_dataset = ActiveLearningDataset(samples=test_dataset.data,
                                                targets=test_dataset.targets,
                                                dataloader_config=loader_config)
    test_active_dataset.add_label(test_active_dataset.get_unlabeled_index())
    
    return train_active_dataset, test_active_dataset
