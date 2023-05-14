import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.layers import DepthwiseConv2D

import sys
import numpy as np

import tensorflow_addons as tfa #pip install tensorflow-addons

import os
os.environ['top'] = '../../../'
sys.path.append(os.path.join(os.environ['top']))
from models.tensorflow.utils import signal_processing_layers


Smoothing = signal_processing_layers.Smoothing


'''
idea is very simple
following andrew ngs coursera conv networks video
using convolution to implement sliding windows
here the convolution is designed specifically for a (64x650x1) lidar frame input
so that the entire image is divided into horizontal segments
in all segment the model simulataneously predicts whether the feature is there or not 0/1
and what is the percentage distance of the center of the feature from the start of the window to end of window (continuous value between 0 and 1)
so that model trainer has to do simulatenous classification and regression
'''
class Network(object): #31,000 parameters only
  def __init__(self, n_channels=1, height=64, width=650, initializer = 'he_normal', lr = 0.0005, use_median_layer = True):
    self.width = width
    self.height = height
    self.n_channels = n_channels
    self.lr = lr
    self.use_median_layer = use_median_layer

    self.input_size = (height*width*n_channels,)
    self.initializer = initializer
    
    self.classif_layer_name = "final_class"
    self.reg_layer_name = "box_dims"

    self.smoothing = Smoothing(input_shape = (height,width,n_channels))

    self.model = self.get_model()
    

  def classif_module(self, tensor_input):
    
    x = Conv2D(1, (2,2), strides=(1,1), activation = 'sigmoid', name = self.classif_layer_name, use_bias=False) (tensor_input)
    print("shape of x ",x.shape) #(None, 1, 18, 2)
    return x

  def reg_module(self, tensor_input):
    x = Conv2D(32, (2,2), strides=(1,1), activation = 'relu', use_bias=False) (tensor_input)
    print("shape of x ",x.shape) #(None, 1, 18, 64)
    x = Conv2D(4, (1,1), strides=(1,1), activation = 'sigmoid', name = self.reg_layer_name, use_bias=False, padding='same') (x)
    print("shape of x ",x.shape) #(None, 1, 18, 2)
    return x

  def multiscale(self, tensor_input):
    x0 = Conv2D(1, (3,3), dilation_rate=8, activation = 'relu', use_bias=False, padding='same') (tensor_input)
    print("shape of x ",x0.shape) #(None, 4, 20, 16) -- divides the image into 4x20=80 regions for 16 different filter combinations

    x1 = Conv2D(1, (3,3), dilation_rate=4, activation = 'relu', use_bias=False, padding='same') (tensor_input)
    print("shape of x ",x1.shape) #(None, 4, 20, 16) -- divides the image into 4x20=80 regions for 16 different filter combinations

    x2 = Conv2D(1, (3,3), dilation_rate=2, activation = 'relu', use_bias=False, padding='same') (tensor_input)
    print("shape of x ",x2.shape) #(None, 4, 20, 16) -- divides the image into 4x20=80 regions for 16 different filter combinations

    x3 = Conv2D(1, (3,3), activation = 'relu', use_bias=False, padding='same') (tensor_input)
    print("shape of x ",x3.shape) #(None, 4, 20, 16) -- divides the image into 4x20=80 regions for 16 different filter combinations

    x4 = Conv2D(1, (1,1), strides=(1,1), activation = 'relu', use_bias=False, padding='same') (tensor_input)
    print("shape of x ",x4.shape) #(None, 4, 20, 16) -- divides the image into 4x20=80 regions for 16 different filter combinations

    res = tf.keras.layers.concatenate([x0,x1,x2,x3,x4], axis=3)
    res = Conv2D(1, (1,1), strides=(1,1), activation = 'relu', use_bias=False, padding='same') (res)

    return res

  def fire_module(self, tensor_input):
    x = Conv2D(16, (1,1), strides=(1,1), padding='valid',activation = 'relu', use_bias=False) (tensor_input)
    xl = Conv2D(64, (1,1), strides=(1,1), padding='valid',activation = 'relu', use_bias=False) (x)
    xr = Conv2D(64, (3,3), strides=(1,1), padding='same',activation = 'relu', use_bias=False) (x)
    cx = tf.keras.layers.concatenate([xl,xr], axis=3)
    return cx
  
  def backbone(self, tensor_input):
    #x = self.smoothing.gaussian_layer(tensor_input)
    if self.use_median_layer:
      x = self.smoothing.median_layer(tensor_input)
      x = tf.reshape(x, [-1,self.height,self.width,self.n_channels])
      #x = median_layer()(tensor_input[0])
      print("median layer x shape ",x.shape)
      x = self.multiscale(x)
    else:
      x = self.multiscale(tensor_input)
    

    x = Conv2D(16, (8,32), strides=(8,10), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 4, 20, 16) -- divides the image into 4x20=80 regions for 16 different filter combinations

    x = Conv2D(32, (3,3), strides=(1,1), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 3, 19, 32)

    x = self.fire_module(x)
    
    print("shape after fire module ",x.shape)
    x = MaxPool2D()(x)
    x = self.fire_module(x)
    
    print("shape after 2nd fire module ",x.shape)

    x = Conv2D(16, (2,2), strides=(1,1), activation = 'relu', use_bias=False) (x)

    return x


    '''
    #prev stuff
    x = Conv2D(16, (3,3), strides=(1,1), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 4, 20, 16) -- divides the image into 4x20=80 regions for 16 different filter combinations
    x = Conv2D(32, (2,2), strides=(1,2), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 3, 19, 32)
    x = Conv2D(32, (2,2), strides=(1,1), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 2, 18, 32)
    x = Conv2D(64, (2,2), strides=(1,1), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 1, 18, 64)
    x = Conv2D(64, (2,2), strides=(1,1), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 1, 18, 64)


    return x
    '''
  

  
  def get_model(self):
    print("constructing sliding conv model ")
    input_shape = (self.height,self.width,self.n_channels)
    print("input shape ",input_shape)
    """
        Model architecture
    """
    
    
    # Define the tensors for the two input images
    input_array = Input(input_shape)

    #dont understand why use_bias=False does much better than use_bias=False

    '''
    x = Conv2D(16, (16,65), strides=(16,30), activation = 'relu', use_bias=False) (input_array)
    print("shape of x ",x.shape) #(None, 4, 20, 16) -- divides the image into 4x20=80 regions for 16 different filter combinations
    x = Conv2D(32, (2,2), strides=(1,1), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 3, 19, 32)
    x = Conv2D(32, (2,2), strides=(1,1), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 2, 18, 32)
    x = Conv2D(64, (2,1), strides=(1,1), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 1, 18, 64)
    x = Conv2D(64, (1,1), strides=(1,1), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 1, 18, 64)
    y = Conv2D(5, (1,1), strides=(1,1), activation = 'relu', use_bias=False) (x)
    print("shape of x ",x.shape) #(None, 1, 18, 2)
    '''


    x = self.backbone(input_array)
    c = self.classif_module(x)
    r = self.reg_module(x)


    #x = Flatten()(x)
    #print("shape of x ",x.shape) #256,256

    net = tf.keras.Model(inputs=input_array,outputs=[c,r])
    #siamese_net.compile(optimizer=Adam(learning_rate=self.lr), loss = tf.keras.losses.KLDivergence())
    #siamese_net.compile(optimizer=Adam(learning_rate=self.lr), loss = tf.keras.losses.MeanAbsoluteError())
    
    

    return net



    

if __name__ == '__main__':


    m1 = Network(width=512)
    print(m1.model.summary())


    '''
    print("checking smoothing network ")
    smoothing = Smoothing()
    smoothing.check_sample_out("sample.png")
    '''








