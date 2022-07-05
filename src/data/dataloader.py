# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")

from tensorflow.keras.utils import Sequence
import albumentations as A
import tensorflow as tf
from glob import glob
from config import *
from utils import *
import numpy as np
import cv2
import os

images_train  = glob(os.path.join(TRAIN_IMAGES, "/*"))
masks_train  = glob(os.path.join(TRAIN_MASKS, "/*"))
images_test  = glob(os.path.join(TEST_IMAGES, "/*"))
masks_test  = glob(os.path.join(TEST_MASKS, "/*"))

images_train = make_list(images_train)
masks_train = make_list(masks_train)
images_test = make_list(images_test)
masks_test = make_list(masks_test)

# Data Generator Class
class CoNSePDataset(Sequence):
    def _init_(self, img_paths = None, mask_paths = None, aug = True):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = A.Compose([      
            A.VerticalFlip(p = 0.5),
            A.HorizontalFlip(p = 0.4),
            A.Transpose(p = 0.5),          
            A.RandomRotate90(p = 0.5),
            A.GridDistortion(p = 0.3),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.RandomBrightnessContrast(p=0.4),    
            A.RandomGamma(p=0.5)
        ])
        assert len(self.img_paths) == len(self.mask_paths)
        self.images = len(self.img_paths) #list all the files present in that folder...
  
    def _len_(self):
        return len(self.img_paths) #length of dataset
  
    def _getitem_(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        if self.aug:
            augment = self.transform(image = image, mask = mask)
            image = augment['image']
            mask = augment['mask']

        image = image.astype(np.float32)
        image = image/255.0
        
        mask = mask.astype(np.float32)
        mask = rgb_to_onehot(mask)
        mask = np.expand_dims(mask, axis = 0)
        return np.expand_dims(image, axis = 0), mask

class CPMKUMARDataset(Sequence):
  def _init_(self, img_paths = None, mask_paths = None, aug = False):
    self.img_paths = img_paths
    self.mask_paths = mask_paths
    self.aug = aug
    self.transform = aug
    assert len(self.img_paths) == len(self.mask_paths)
    self.images = len(self.img_paths) #list all the files present in that folder...
  
  def _len_(self):
    return len(self.img_paths) #length of dataset
  
  def _getitem_(self, index):
    img_path = self.img_paths[index]
    mask_path = self.mask_paths[index]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)

    if self.aug:
        augment = self.transform(image = image, mask=mask)
        image = augment['image']
        mask = augment['mask']

    image = image.astype(np.float32)
    image = image/255.0
    
    mask = mask.astype(np.float32)
    mask = rgb_to_onehot(mask)
    mask = mask.astype(np.float32)
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis = 0)

    return np.expand_dims(image, axis = 0), mask

# this function we call during traing to get set of dataloader i.e train loader and val_loader
def getData(dataset):
    if dataset.lower() == "consep":
        train_ds = CoNSePDataset(
                img_paths = images_train,
                mask_paths = masks_train,
                aug = True
            )

        test_ds = CoNSePDataset(
                img_paths = images_test,
                mask_paths = masks_test,
                aug = False)
        
        return train_ds, test_ds

    elif (dataset.lower() == "cpm") or (dataset.lower() == "kumar"):
        train_ds = CPMKUMARDataset(
                img_paths = images_train,
                mask_paths = masks_train,
                aug = True
            )

        test_ds = CPMKUMARDataset(
                img_paths = images_test,
                mask_paths = masks_test,
                aug = False)
        return train_ds, test_ds