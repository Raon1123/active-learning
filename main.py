import argparse
import os

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
    cfg = load_config(args.config)
    
    
    
if __name__ == "__main__":
    main()