# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../src/data/")

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from dataloader import *
from tqdm import tqdm
from config import *
from utils import *
from loss import *
import numpy as np
import argparse
import logging
import os

def main():
    """ testing performance of model. 
    """
    train_ds, test_ds = getData(DATASET)
    model = tf.keras.models.load_model(os.path.join(WEIGHTS_DIR, 'best_model.h5'), custom_objects = {"diceCoef": diceCoef, "bceDiceLoss": bceDiceLoss})
    loader = test_ds
    y_true = []
    y_pred = []
    y_true_inst = []
    y_pred_inst = []

    mPQ_all = []
    bPQ_all = []
    aji_all = []

    for i, data in enumerate(loader):
        x = data[0]
        yhat = model.predict(x)
        y = data[1]
        y_inst = preocess_utils(1 - y[0, :, :, 0])
        y_true_inst.append(y_inst)
        yhat = ((yhat[0,:,:,:])>0.5).astype('float32')
        yhat_inst = process_utils(1 - yhat[0,:,:,0])
        y_pred_inst.append(yhat_inst)
        y_true.append(y)
        y_pred.append(yhat)

    true = np.array(y_true)
    pred = np.array(y_pred)
    
    true_inst = np.load('inst_true.npy', mmap_mode='r')
    pred_inst = np.load('inst_pred.npy', mmap_mode='r')

    for i in range(true.shape[0]):
        pq = []
        aji = []

        for j in range(5):
            pred_tmp = pred[i,:,:,j]
            pred_tmp = pred_tmp.astype('float32')
            true_tmp = true[i,:,:,j]
            true_tmp = true_tmp.astype('float32')

            if len(np.unique(true_tmp)) == 1:
                pq_tmp = np.nan # if ground truth is empty for that class, skip from calculation
            else:
                pq_tmp = get_fast_pq(true_tmp, pred_tmp, 0.3)
                aji_temp = get_fast_aji(true_tmp, pred_tmp)
                
                print(pq_tmp) # compute PQ

            pq.append(pq_tmp)
            aji.append(aji_temp)

        mPQ_all.append(pq)
        aji_all.append(aji)


    logging.info('-' * 40)
    logging.info('Average mPQ:{}'.format(np.nanmean(mPQ_all)))
    logging.info('Average AJI:{}'.format(np.nanmean(aji_all)))
    logging.info('Average bPQ:{}'.format(np.nanmean(mPQ_all)))
    logging.info('Average DQ:{}'.format(np.nanmean(mPQ_all)))
    logging.info('Average SQ:{}'.format(np.nanmean(mPQ_all)))
    logging.info('Average DICE:{}'.format(np.nanmean(mPQ_all)))
    logging.info('Average Precision:{}'.format(np.nanmean(mPQ_all)))
    logging.info('Average Recall:{}'.format(np.nanmean(mPQ_all)))
    logging.info('Average F1-score:{}'.format(np.nanmean(mPQ_all)))
    

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    main()