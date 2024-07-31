import argparse
from train import main as train_main
from inference import main as inference_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or Inference for Keypoint Detection')
    parser.add_argument('mode', choices=['train', 'inference'], help='Mode to run: train or inference')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_main()
    elif args.mode == 'inference':
        inference_main()
