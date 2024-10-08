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
    
    train_dataset = data_dict[dataset_name](root=config['DATASET']['root'], 
                                            train=True, download=True)
    test_dataset = data_dict[dataset_name](root=config['DATASET']['root'],
                                             train=False, download=True)
    
    train_active_dataset = ActiveLearningDataset(samples=train_dataset.data,
                                                    targets=train_dataset.targets)
    test_active_dataset = ActiveLearningDataset(samples=test_dataset.data,
                                                targets=test_dataset.targets)
    
    return train_active_dataset, test_active_dataset


def get_dataloader(train_dataset, test_dataset, config):
    loader_config = config['DATASET']['dataloader']
    
    loader_config['shuffle'] = True
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                **loader_config)
    
    loader_config['shuffle'] = False
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                **loader_config)
    
    return train_loader, test_loader