import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
import sys

'''
def loss(y_true, y_pred):
  mse =  tf.keras.losses.MeanSquaredError()
  return mse(y_true, y_pred)
'''

def loss(y_true, y_pred):
  classifs_pred = y_pred[:,0,:,0]
  classifs_true = y_true[:,0,:,0]

  #print(classifs_true,classifs_pred)

  regs_pred = y_pred[:,0,:,1:]
  regs_true = y_true[:,0,:,1:]

  #print(regs_true,regs_pred)

  mse =  tf.keras.losses.MeanSquaredError()
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

  mse_l = 0.5*mse(regs_true, regs_pred)
  bce_l = 0.5*bce(classifs_true, classifs_pred)

  full_mse = mse(y_true,y_pred)

  #scales of mse and bce are widly different so trying to make same
  comb_loss = 0.5*bce_l + 0.5*800*mse_l

  return comb_loss


def class_loss(y_true, y_pred):
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False) #because model last layer is sigmoid already
  #bce =  tf.keras.losses.MeanSquaredError() #sometimes substituting bce with mse in smaller networks with only 1 class gives better results
  return bce(y_true, y_pred)


def reg_loss(y_true, y_pred):
  mse =  tf.keras.losses.MeanSquaredError()
  return mse(y_true, y_pred)

  

  