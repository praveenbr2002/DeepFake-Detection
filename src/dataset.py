import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from albumentations import (
    Compose, FancyPCA, GaussianBlur, GaussNoise, HorizontalFlip,
    HueSaturationValue, ImageCompression, OneOf, PadIfNeeded, 
    RandomBrightnessContrast, Resize, ShiftScaleRotate, ToGray
)

class CelebDFDataset(Dataset):

    def __init__(self, data, img_size, method, normalization, augmentations):
        self.data = data
        self.img_size = img_size
        self.method = method
        self.normalization = normalization
        self.augmentations = augmentations

    def __getitem__(self, idx):

        image_row = self.data.iloc[idx]
        label = image_row.loc['label']

        images = []
        image = image_row.loc['original']

        for i in range(20):

            if label == 1:
                img_path = os.path.join(image + '_' + str(i) + '.jpg')
            else:
                img_path = os.path.join(image + '_' + str(i) + '.jpg')
            try:
                img = cv2.imread(img_path)
            except:
                print(img_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.augmentations:
                img = self.augmentations(image=img)['image']
            else:
                augmentations = Resize(width=self.img_size, height=self.img_size)
                img = augmentations(image=img)['image']

            img = torch.tensor(img).permute(2, 0, 1)
            img = img.float() / 255.0

            if self.normalization == "xception":
                transform = transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            
            elif self.normalization == "imagenet":
                transform = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
            img = transform(img)
            
            images.append(img.numpy())
        
        return np.array(images), label

    def __len__(self):
        return len(self.data)