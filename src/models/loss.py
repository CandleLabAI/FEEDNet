# -*- coding: utf-8 -*-
from tensorflow.keras import backend as  K
import tensorflow.keras.losses.binary_crossentropy
import tensorflow_addons as tfa
import tensorflow as tf

def diceCoef(y_true, y_pred):   
    smooth = 1.0
    y_true_f = K.flatten(y_true)    
    y_pred_f = K.flatten(y_pred)    
    intersection = K.sum(y_true_f * y_pred_f)    
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def diceCoefLoss(y_true, y_pred):
    return 1.-diceCoef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + diceCoefLoss(y_true, y_pred)
    return loss
