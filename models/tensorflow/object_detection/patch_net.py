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
    
    x = Conv2D(10, (1,1), strides=(1,1), padding='same',activation = 'relu', use_bias=False) (tensor_input)
    print("classif module x shape ",x.shape)
    #sys.exit(0)


    xc = tf.keras.layers.GlobalAveragePooling2D()(x)
    xc = tf.reshape(xc, [-1,1,10,1])
    xc = Conv2D(1, (1,1), strides=(1,1), padding='same',activation = 'sigmoid', use_bias=False, name = self.classif_layer_name) (xc)
    print("x shape after global average maxpool ",xc.shape)
    return xc

  def reg_module(self, tensor_input):
    y = Conv2D(40, (2,2), strides=(1,1), activation = 'relu', use_bias=False) (tensor_input)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = tf.reshape(y,[-1,1,10,4])
    y = Conv2D(4, (1,1), strides=(1,1), padding='same',activation = 'sigmoid', use_bias=False, name = self.reg_layer_name) (y)
    print("y shape after global average maxpool ",y.shape)
    return y



  
  def special(self, tensor_input):
    xd1 = DepthwiseConv2D((3,3), padding='same', activation = 'relu', use_bias=False) (tensor_input)
    print("shape of x ",xd1.shape) #(None, 2, 18, 32)
    xd2 = DepthwiseConv2D((2,2), padding='same', activation = 'relu', use_bias=False) (tensor_input)
    print("shape of x ",xd2.shape) #(None, 2, 18, 32)
    xd3 = DepthwiseConv2D((1,1), padding='same', activation = 'relu', use_bias=False) (tensor_input)
    print("shape of x ",xd3.shape) #(None, 2, 18, 32)

    y = tf.keras.layers.add([xd1, xd2, xd3])
    y = tf.keras.layers.BatchNormalization()(y)
    return y

  
  def backbone(self, tensor_input): #very small one
    if self.use_median_layer:
      #x = self.smoothing.gaussian_layer(tensor_input)
      x = self.smoothing.median_layer(tensor_input)
      x = tf.reshape(x, [-1,self.height,self.width,self.n_channels])
      #x = median_layer()(tensor_input[0])
      print("median layer x shape ",x.shape)
      #x = self.multiscale(x)
    else:
      x = tensor_input

    #x = Conv2D(10, (30,20), strides=(1,20), activation = 'relu', use_bias=False) (x)
    #print("x shape after special conv ",x.shape)

    x = tf.image.extract_patches(x, [1, 64, 65, 1], [1, 64, 65, 1], [1,1,1,1], 'VALID')
    x = tf.reshape(x,[-1,64,65,10]) #reshape the patches to their patch size and stack all of them as different channels 
    print("shape of x after extract patches ",x.shape) #(None, 4, 20, 16) -- divides the image into 4x20=80 regions for 16 different filter combinations



    #x = self.special(x)
    #x = self.special(x)

    #x = Conv2D(100, (2,2), strides=(1,1), activation = 'relu', use_bias=False) (x)
    #x = Conv2D(50, (1,1), strides=(1,1), padding='same',activation = 'relu', use_bias=False) (x)

    x = Conv2D(16, (1,1), strides=(1,1), padding='same',activation = 'relu', use_bias=False) (x)
    
    xl = Conv2D(64, (1,1), strides=(1,1), padding='same',activation = 'relu', use_bias=False) (x)
    
    xr = Conv2D(64, (3,3), strides=(1,1), padding='same',activation = 'relu', use_bias=False) (x)
    
    cx = tf.keras.layers.concatenate([xl,xr], axis=3)
    x = MaxPool2D()(cx)


    

    return x


  

  
  def get_model(self):
    print("constructing sliding conv model ")
    input_shape = (self.height,self.width,self.n_channels)
    print("input shape ",input_shape)
    """
        Model architecture
    """
    
    
    # Define the tensors for the two input images
    input_array = Input(input_shape)




    x = self.backbone(input_array)
    c = self.classif_module(x)
    r = self.reg_module(x)

    net = tf.keras.Model(inputs=input_array,outputs=[c,r])
    
    

    return net



    

if __name__ == '__main__':


    m1 = Network()
    print(m1.model.summary())


    print("checking smoothing network ")
    smoothing = Smoothing()
    smoothing.check_sample_out("sample.png")








