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
    correct = 0
    total = 0
    loop = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', acc, epoch)
    return avg_loss

def validate(model, loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(loader, desc=f'Epoch {epoch} [Val]')
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total

    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Accuracy', acc, epoch)

    return avg_loss, acc

def train_model(model, train_loader, val_loader, config):
    device = torch.device(config['training']['device'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    writer = SummaryWriter('runs/disease_classification')
    best_val_loss = float('inf')
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, writer, epoch)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'classifier_best.pth'))
            print("Saved best model")

    writer.close()