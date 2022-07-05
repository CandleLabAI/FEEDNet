# -*- coding: utf-8 -*-
# yet to be modified
import sys
sys.path.append("../src")

from tqdm import tqdm 
from config import *
from utils import *
import numpy as np
import logging
import cv2
import os

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be trained (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('[Info] Generating Images, Masks from raw data')
    imagesDir = "{update_src_image_dir_here}"
    masksDir = "{update_src_mask_dir_here}"
    destImagesDir = "{update_dest_image_dir_here}"
    destMasksDir = "{update_dest_mask_dir_here}"

    if not os.path.exists(destImagesDir):
      os.makedirs(destImagesDir)
    if not os.path.exists(destMasksDir):
      os.makedirs(destMasksDir)

    imagesLst = sorted([os.path.join(imagesDir, image) for image in os.listdir(imagesDir)])
    masksLst = sorted([os.path.join(masksDir, image) for image in os.listdir(masksDir)])

    patchSize = 256
    count = 0
    shapes = {}

    with tqdm(total=len(imagesLst)) as pbar:
        for i, (image, mask) in enumerate(zip(imagesLst, masksLst)):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            msk = cv2.imread(mask)
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)

            for j in range(0, img.shape[0], patchSize):
                for k in range(0, img.shape[1], patchSize):
                    temp = img[j:j+patchSize, k:k+patchSize, :]
                    if temp.shape[0] != patchSize:
                        j = j - (patchSize - temp.shape[0])
                        temp = img[j:j+patchSize, k:k+patchSize, :]
                    if temp.shape[1] != patchSize:
                        k = k - (patchSize - temp.shape[1])
                        temp = img[j:j+patchSize, k:k+patchSize, :]
                    if str(img[j:j+patchSize, k:k+patchSize, :].shape) in shapes:
                        shapes[str(img[j:j+patchSize, k:k+patchSize, :].shape)] += 1
                    else:
                        shapes[str(img[j:j+patchSize, k:k+patchSize, :].shape)] = 1
                    cv2.imwrite(os.path.join(destImagesDir, "image_{}.png".format(count)), cv2.cvtColor(img[j:j+patchSize, k:k+patchSize, :], cv2.COLOR_BGR2RGB))
                    cv2.imwrite(os.path.join(destMasksDir, "image_{}.png".format(count)), cv2.cvtColor(msk[j:j+patchSize, k:k+patchSize, :], cv2.COLOR_BGR2RGB))

                    count += 1
            pbar.update(1)

    print(f"There are total {len(os.listdir(destImagesDir))} images")
    print(f"There are total {len(os.listdir(destMasksDir))} masks")

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    main()
