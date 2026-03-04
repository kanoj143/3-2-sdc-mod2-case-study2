import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from src.evaluate import dice_coefficient, iou_score

def train_one_epoch(model, loader, optimizer, criterion, device, writer, epoch):
    if len(loader) == 0:
        print("Error: Training loader is empty!")
        return 0
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        predictions = model(images)
        loss = criterion(predictions, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        if batch_idx % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(loader) + batch_idx)

    return total_loss / len(loader)

def validate(model, loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    with torch.no_grad():
        loop = tqdm(loader, desc=f'Epoch {epoch} [Val]')
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)
            loss = criterion(predictions, masks)

            total_loss += loss.item()
            dice = dice_coefficient(predictions, masks)
            iou = iou_score(predictions, masks)
            total_dice += dice.item()
            total_iou += iou.item()

            loop.set_postfix(loss=loss.item(), dice=dice.item())

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)

    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Dice', avg_dice, epoch)
    writer.add_scalar('Val/IoU', avg_iou, epoch)

    return avg_loss, avg_dice, avg_iou

def train_model(model, train_loader, val_loader, config):
    device = torch.device(config['training']['device'])
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    writer = SummaryWriter('runs/retina_experiment')
    best_val_loss = float('inf')
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device, writer, epoch)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'unet_best.pth'))
            print("Saved best model")

    writer.close()