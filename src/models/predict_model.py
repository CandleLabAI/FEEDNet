# -*- coding: utf-8 -*-
# usage: python models/predict_model.py --image {path_of_image} 
import sys
sys.path.append("../src")

import matplotlib.pyplot as plt
import tensorflow as tf
from config import *
from utils import *
from loss import *
import numpy as np
import argparse
import logging
import os

def main(args):
    """ predicting given image. 
    """
    model = tf.keras.models.load_model(os.path.join(WEIGHTS_DIR, 'best_model.h5'), custom_objects={"diceCoef": diceCoef, "bceDiceLoss": bceDiceLoss})
    img = cv2.imread(args["image"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)
    pred = model.predict(img).numpy()
    pred_mask = tf.keras.utils.to_categorical(pred.argmax(-1), num_classes = 2)
    pred_mask = onehot_to_rgb(pred_mask)
    pred_mask = process_utils(pred_mask, "seg_nuc")
    pred_mask = remap_label(pred_mask, by_size=True)
    overlaid_pred_mask = visualize_instances(pred_mask, img)
    overlaid_pred_mask = cv2.cvtColor(overlaid_pred_mask, cv2.COLOR_BGR2RGB)
    plt.imsave(os.path.join(INFERENCE_DIR, "{}.png".format(os.path.splitext(os.path.basename(args["image"]))[0])), overlaid_pred_mask)
    logging.info("[Info] Overlay image is saved in Inference directory")

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image in numpy format")
    args = vars(ap.parse_args())

    main(args)