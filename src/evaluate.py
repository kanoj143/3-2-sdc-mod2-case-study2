import torch
import numpy as np

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient per channel and average."""
    pred = (pred > 0.5).float()  # threshold
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU per channel and average."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    total = (pred + target).sum(dim=(2, 3))
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def evaluate_model(model, test_loader, device):
    model.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print(f"Test Dice: {avg_dice:.4f}, Test IoU: {avg_iou:.4f}")
    return avg_dice, avg_iou