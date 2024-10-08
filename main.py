import argparse
import os

from dataset.datatool import (
    get_dataset
)
from model.modeltool import (
    get_model,
    get_optimizer
)
from sampler.random import (
    RandomSampler
)
from toolkits.epochs import run_round
from toolkits.toolkit import (
    is_path,
    load_config
)

def get_args():
    parser = argparse.ArgumentParser(description="Active Learning framework")
    
    parser.add_argument("--config", "-c", 
                        type=is_path, required=True,
                        help="Path to configuration file")
    parser.add_argument("--verbose", "-v",
                        action="store_true",
                        help="Verbose mode")
    
    args = parser.parse_args()
    
    return args


def main():
    args = get_args()
    config = load_config(args.config)
    
    train_dataset, test_dataset = get_dataset(config)
    model = get_model(config).to(config['DEVICE'])
    optimizer = get_optimizer(config, model)
    
    sampler = RandomSampler()
    n_samples = config['SAMPLER']['n_samples']
    
    rounds = config['MODEL']['num_round']
    test_loader = test_dataset.get_dataloader()
     
    for round in range(rounds):
        sample_idx = sampler.sample(train_dataset, n_samples)
        train_dataset.add_label(sample_idx)
        
        train_loader = train_dataset.get_dataloader()
        train_loss, test_loss, test_acc = run_round(train_loader, test_loader, 
                        model, optimizer, config)
        
        print(f"Round {round+1}: Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")
        print(f"Sampled indices: {sample_idx}")
    
if __name__ == "__main__":
    main()