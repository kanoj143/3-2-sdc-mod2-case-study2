import argparse
import os
import torch
from src.data_loader import create_dataloaders
from src.disease_classifier import get_classifier
from src.train import train_model
from src.utils import load_config, create_splits
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def predict(model, image_path, config, device):
    model.eval()
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    input_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        
    disease_classes = config['model']['diseases']
    detected_disease = disease_classes[pred.item()]
    
    print(f"\n--- Analysis Result ---")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Detected Disease: {detected_disease}")
    print(f"Confidence: {conf.item()*100:.2f}%")
    print(f"------------------------\n")

def main():
    parser = argparse.ArgumentParser(description='Retinal Disease Detection')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'split'], default='train')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config['training']['device'])

    if args.mode == 'split':
        create_splits(config['data']['raw_images_dir'], config['data']['split_dir'])
        return

    # Initialize model
    model = get_classifier(config)

    if args.mode == 'train':
        train_loader, val_loader, _ = create_dataloaders(config)
        train_model(model, train_loader, val_loader, config)

    elif args.mode == 'predict':
        if not args.image:
            print("Error: Please provide an image path with --image")
            return
        
        weights_path = os.path.join(config['training']['save_dir'], 'classifier_best.pth')
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            print("Warning: Model weights not found. Using uninitialized model for demonstration.")
            
        model.to(device)
        predict(model, args.image, config, device)

if __name__ == '__main__':
    main()