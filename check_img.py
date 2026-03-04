import cv2
import os

img_path = r"e:\ASDC-2025\3-2\Soft-computing-ML\Soft-Computing-Projects\Module-2\case-study2\data\raw\images\Screenshot 2026-03-03 113230.png"
if os.path.exists(img_path):
    img = cv2.imread(img_path)
    if img is not None:
        print(f"Shape: {img.shape}")
    else:
        print("Failed to load image")
else:
    print("File not found")
