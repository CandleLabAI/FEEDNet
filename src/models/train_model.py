# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../src/data/")

import tensorflow as tf
from dataloader import *
from network import *
from loss import *
from config import *
from utils import *
import logging
import os

def main():
    """ training WaferSegClassNet model 
    """
    logger = logging.getLogger(__name__)
    logger.info("[Info] Getting DataLoader")
    
    train_ds, test_ds = getData(DATASET)
    logger.info("[Info] Creating Network")
    model = getModel()
    logger.info("[Info] Summary of contrastive model \n")
    logger.info(model.summary())

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), 
                  loss = {'segmentation': bce_dice_loss},
                  metrics = {'segmentation': ['accuracy', diceCoef]},
                  run_eagerly = True
                  )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(WEIGHTS_DIR, 'best_model.h5'), monitor = 'val_diceCoef', mode="max", verbose = 1, save_best_only = True, save_weights_only = False),
    ]

    model.fit(train_ds, validation_data = test_ds, epochs = EPOCHS, callbacks = callbacks)  
    logger.info("[Info] Model training is Finished")

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    main()
