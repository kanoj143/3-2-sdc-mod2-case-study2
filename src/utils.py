import yaml
import os

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_splits(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test CSV files from available images."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    image_files = [f.replace('.png', '') for f in os.listdir(data_dir) if f.endswith('.png')]
    np.random.seed(seed)
    train_val, test = train_test_split(image_files, test_size=1-train_ratio-val_ratio, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=seed)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'filename': train}).to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    pd.DataFrame({'filename': val}).to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    pd.DataFrame({'filename': test}).to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Split created: {len(train)} train, {len(val)} val, {len(test)} test")