import os
import warnings
import multiprocessing
from pathlib import Path
from urllib.parse import urlparse
#import av
import datasets
import numpy as np
import random
import torch
import torch.nn as nn
import torchaudio.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from typing import Dict, List
import torchaudio
warnings.filterwarnings("ignore")

import random
try:
    multiprocessing.set_start_method('fork', force=True)
except:
    multiprocessing.set_start_method('spawn', force=True)
import gc
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

class LocalCaptionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
            #transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)), 
        ])
        
        self.clean_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        self.error_tensor = torch.zeros((3, 224, 224))
        self.image_files = []
        for subdir in self.root_dir.iterdir():
            if subdir.is_dir():
                self.image_files.extend(list(subdir.glob("*.jpg")))
        print(f"Found {len(self.image_files)} images in {self.root_dir}")
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        txt_path = img_path.with_suffix('.txt')
        
        try:
            with Image.open(img_path) as image:
                image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
            with open(txt_path, 'r') as f:
                caption = f.read().strip()
                
            return image, caption
            
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.error_tensor, ""


if __name__ == "__main__":
    print("Testing LocalCaptionDataset...")
    dataset = LocalCaptionDataset("/home/cis/cc3m")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    print("\nTesting batch loading...")
    for batch_idx, (images, captions) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")
        print(f"Image batch shape: {images.shape}")  # Should be [4, 3, 224, 224]
        print(f"Sample caption: {captions[0]}")
        break
    import tqdm
    
        