import argparse
import os
import torch
from src.data_loader import create_dataloaders
from src.model import UNet
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import load_config, create_splits

def main():
    parser = argparse.ArgumentParser(description='Retinal Disease Detection')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'split'], default='train')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == 'split':
        create_splits(config['data']['raw_images_dir'], config['data']['split_dir'])
        return

    # Prepare dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Initialize model
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        features=config['model']['features']
    )

    if args.mode == 'train':
        train_model(model, train_loader, val_loader, config)

    elif args.mode == 'evaluate':
        device = torch.device(config['training']['device'])
        weights_path = os.path.join(config['training']['save_dir'], 'unet_best.pth')
        if not os.path.exists(weights_path):
            print(f"Error: Model weights not found at {weights_path}. Please train the model first.")
            return
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()