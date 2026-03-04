import torch
import cv2
import numpy as np
from src.model import UNet
from src.utils import load_config
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(checkpoint_path, config):
    model = UNet(in_channels=config['model']['in_channels'],
                 out_channels=config['model']['out_channels'])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image_path, config):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0)  # add batch dim
    return image_tensor, image

def postprocess_mask(mask_tensor, original_image):
    # mask_tensor: (1, 2, H, W) after sigmoid
    mask = mask_tensor.squeeze(0).cpu().detach().numpy()  # (2, H, W)
    mask = np.transpose(mask, (1, 2, 0))  # (H, W, 2)
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask  # each channel as binary mask

def infer(image_path, model, config, device='cpu'):
    image_tensor, original = preprocess_image(image_path, config)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    mask = postprocess_mask(output, original)
    return mask, original

def visualize_overlay(image, mask, alpha=0.5):
    """Create overlay: vessels in green, microaneurysms in red."""
    overlay = image.copy()
    # Vessels (channel 0) -> green
    overlay[:,:,1] = np.where(mask[:,:,0] > 0, 255, overlay[:,:,1])
    # Microaneurysms (channel 1) -> red
    overlay[:,:,0] = np.where(mask[:,:,1] > 0, 255, overlay[:,:,0])
    blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    return blended