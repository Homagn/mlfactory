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



class Smoothing(object):
  #ref- https://stackoverflow.com/questions/55643675/how-do-i-implement-gaussian-blurring-layer-in-keras
  def __init__(self, shape = (10,10), sigma = 3.2, input_shape = (3,3,1)): #input_shape = (height, width, channels) of input image
    self.shape = shape
    self.sigma = sigma
    self.input_shape = input_shape
    
    self.gaussian_layer = self.build_layer_gaussian()
    self.median_layer = self.build_layer_median()

  def gauss2D(self):

      m,n = [(ss-1.)/2. for ss in self.shape]
      y,x = np.ogrid[-m:m+1,-n:n+1]
      h = np.exp( -(x*x + y*y) / (2.*self.sigma*self.sigma) )
      h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
      sumh = h.sum()
      if sumh != 0:
          h /= sumh
      return h

  def build_layer_gaussian(self, shape_override = -1, sigma_override = -1):
      if shape_override!=-1:
        self.shape = shape_override
      if sigma_override!=-1:
        self.sigma = sigma_override
      kernel_size = self.shape[0]
      kernel_weights = self.gauss2D()
      
      
      in_channels = 1  # the number of input channels
      kernel_weights = np.expand_dims(kernel_weights, axis=-1)
      kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1) # apply the same filter on all the input channels
      kernel_weights = np.expand_dims(kernel_weights, axis=-1)  # for shape compatibility reasons
      
      
      inp = Input(shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2]))
      g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')(inp)
      model_network = Model(inputs=inp, outputs=g_layer)
      model_network.layers[1].set_weights([kernel_weights])
      model_network.trainable= False #can be applied to a given layer only as well
          
      return model_network

  def build_layer_median(self):
    

    class median_layer(tf.keras.layers.Layer):
      def __init__(self, batch_size = 1):
          super(median_layer, self).__init__()
          self.batch_size = batch_size
          self.elems = tf.range(0,batch_size,1)
          
      def call(self, inputs):
          
          layers = []
          for i in range(self.batch_size):
            layers.append(tfa.image.median_filter2d(inputs[i], filter_shape=11))
          return tf.keras.layers.concatenate(layers, axis=0)
          
          #return tf.map_fn(fn=lambda i: tfa.image.median_filter2d(inputs[i], filter_shape=11), elems = self.elems)



    

    inp = Input(shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2]))
    median = median_layer()(inp)

    model_network = Model(inputs=inp, outputs=median)
    model_network.trainable= False #can be applied to a given layer only as well
          
    return model_network

  def build_layer_log(self): #laplacian of gaussian
    #ref - https://stackoverflow.com/questions/64342985/compute-the-laplacian-of-gaussian-using-tensorflow
    pass

  

  def check_sample_out(self, image_path):
    import cv2
    
    sample = cv2.imread(image_path)
    w,h = sample.shape[0], sample.shape[1]
    sample = sample[:,:,0]
    cv2.imshow("loaded sample ",sample)
    cv2.waitKey(0)

    self.input_shape = (sample.shape[0],sample.shape[1],1)
    sg1 = self.build_layer_gaussian(shape_override = (15,15), sigma_override = 1.2)
    sg2 = self.build_layer_gaussian(shape_override = (3,3), sigma_override = 3.6)
    
    sm = self.build_layer_median()


    sample = sample/255.0
    sample = sample.reshape((sample.shape[0],sample.shape[1],1))
    x = tf.convert_to_tensor(np.array([sample]), dtype=tf.float32)
    print("x shape ",x.shape)

    
    sample_out = sg1(x)
    sample_out = sg2(sample_out)
    so = np.array(sample_out[0])
    cv2.imshow("sample out gaussian filter",so)
    cv2.waitKey(0)
    

    sample_out = sm(x)
    #sample_out = self.median_filter(x)
    so = np.array(sample_out)
    cv2.imshow("sample out median filter",so)
    cv2.waitKey(0)

    
