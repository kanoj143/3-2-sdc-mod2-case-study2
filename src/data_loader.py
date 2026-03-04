import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.utils import load_config

class RetinaDataset(Dataset):
    def __init__(self, images_dir, labels_df, file_list, transform=None):
        """
        Args:
            images_dir: path to input images
            labels_df: dataframe with 'filename' and 'disease_idx'
            file_list: list of image filenames (without extension)
            transform: albumentations transforms
        """
        self.images_dir = images_dir
        self.labels_df = labels_df.set_index('filename')
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.images_dir, img_name + ".png")
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels_df.loc[img_name, 'disease_idx']
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(phase, config):
    """Return train/val transforms based on config."""
    aug_config = config['augmentation']
    if phase == 'train':
        transforms = [A.Resize(224, 224)] # ResNet typical input size
        if aug_config['horizontal_flip']:
            transforms.append(A.HorizontalFlip(p=0.5))
        if aug_config['vertical_flip']:
            transforms.append(A.VerticalFlip(p=0.5))
        if aug_config['rotation'] > 0:
            transforms.append(A.Rotate(limit=aug_config['rotation'], p=0.5))
        if aug_config['brightness_contrast']:
            transforms.append(A.RandomBrightnessContrast(p=0.2))
        transforms.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        return A.Compose(transforms)
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def create_dataloaders(config):
    """Create train, val, test dataloaders for classification."""
    split_dir = config['data']['split_dir']
    labels_file = config['data']['labels_file']
    
    labels_df = pd.read_csv(labels_file)
    train_df = pd.read_csv(os.path.join(split_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(split_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(split_dir, 'test.csv'))

    train_files = train_df['filename'].tolist()
    val_files = val_df['filename'].tolist()
    test_files = test_df['filename'].tolist()

    train_dataset = RetinaDataset(
        images_dir=config['data']['raw_images_dir'],
        labels_df=labels_df,
        file_list=train_files,
        transform=get_transforms('train', config)
    )
    val_dataset = RetinaDataset(
        images_dir=config['data']['raw_images_dir'],
        labels_df=labels_df,
        file_list=val_files,
        transform=get_transforms('val', config)
    )
    test_dataset = RetinaDataset(
        images_dir=config['data']['raw_images_dir'],
        labels_df=labels_df,
        file_list=test_files,
        transform=get_transforms('val', config)
    )

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=config['training']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                             shuffle=False, num_workers=config['training']['num_workers'])

    return train_loader, val_loader, test_loader