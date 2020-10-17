import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import natsort
from PIL import Image

#torch.from_numpy(masks).float()
class LaneDataset(Dataset):
    def __init__(self, image_path, mask_path, transform = None):
        self.image_path = image_path
        self.mask_path = mask_path
        all_images = os.listdir(self.image_path)
        all_masks = os.listdir(self.mask_path)
        self.total_images = natsort.natsorted(all_images)
        self.total_masks = natsort.natsorted(all_masks)
        self.transform  = transform

    def __getitem__(self, index):
        image_loc = os.path.join(self.image_path, self.total_images[index])
        image = Image.open(image_loc)
        image = self.transform(image)
        mask_loc = os.path.join(self.mask_path, self.total_masks[index])
        mask = Image.open(mask_loc)
        mask = self.transform(mask)
        return image, mask 
        
    def __len__(self):
        return len(self.total_images)
    
