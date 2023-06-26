import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class PneumothoraxImageDataset(Dataset):
    def __init__(self, images, targets, masks, transform, mask_transform):
        self.masks = masks
        self.images = images
        self.targets = targets
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.asarray(Image.open(self.images[idx]).convert('RGB'))
        mask = np.asarray(Image.open(self.masks[idx]).convert('RGB'))
        label = self.targets[idx]
        if self.transform:
          image = self.transform(image)
        if self.mask_transform:
          mask = self.mask_transform(mask)
        return image, mask, label
