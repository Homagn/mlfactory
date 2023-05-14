import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential,Model
import numpy as np




class median_layer(tf.keras.layers.Layer):
  def __init__(self, k, h, w):
      super(median_layer, self).__init__()
      #hardcoding the index locations based on the filter size and the image size
      self.k = k
      self.w = w
      self.h = h

  def call(self, inputs):
      img = tf.reshape(inputs, [self.h,self.w]  )

      patches = tf.image.extract_patches(inputs, [1, self.k, self.k, 1], [1, 1, 1, 1], 4*[1], 'VALID')
      #patches = tf.reshape(patches,[3,3,4])
      medians = tf.math.top_k(patches, k=5)[0][:,:,:,-1] #k=5 is specific to filter size = 3
      updates = tf.reshape(medians, [-1])
      indices = tf.repeat([5], repeats = [(self.w-2)*(self.h-2)], axis=0)

      #need a way to find aranged indices properly
      image_ordering = tf.range(1,(self.w*self.h)+1,1)
      image_ordering_r = tf.reshape(image_ordering,[self.w,self.h])

      image_ordering_s = image_ordering_r[1:-1,1:-1]
      image_ordering_sf = tf.reshape(image_ordering_s, [(self.w-2)*(self.h-2)])


      #aranged_indices = tf.range(1,65,1)
      aranged_indices = image_ordering_sf

      coord_indices = tf.unravel_index(indices= aranged_indices, dims=[self.w, self.h])
      r_coord_indices = tf.stack([coord_indices[0],coord_indices[1]], axis=1)
      
      result = tf.tensor_scatter_nd_update(img, r_coord_indices, updates)

      return r_coord_indices


'''

class median_layer(tf.keras.layers.Layer):
  def __init__(self, k, h, w):
      super(median_layer, self).__init__()
      #hardcoding the index locations based on the filter size and the image size
      self.k = k
      self.w = w
      self.h = h

  def call(self, inputs):
      img = tf.reshape(inputs, [self.h,self.w]  )

      patches = tf.image.extract_patches(inputs, [1, self.k, self.k, 1], [1, 1, 1, 1], 4*[1], 'VALID')
      #patches = tf.reshape(patches,[3,3,4])
      medians = tf.math.top_k(patches, k=60)[0][:,:,:,-1] #k=60 is specific to filter size = 11
      updates = tf.reshape(medians, [-1])
      indices = tf.repeat([60], repeats = [(self.w-10)*(self.h-10)], axis=0)

      #need a way to find aranged indices properly
      image_ordering = tf.range(1,(self.w*self.h)+1,1)
      image_ordering_r = tf.reshape(image_ordering,[self.w,self.h])

      image_ordering_s = image_ordering_r[5:-5,5:-5]
      image_ordering_sf = tf.reshape(image_ordering_s, [(self.w-10)*(self.h-10)])


      #aranged_indices = tf.range(1,65,1)
      aranged_indices = image_ordering_sf

      coord_indices = tf.unravel_index(indices= aranged_indices, dims=[self.w, self.h])
      r_coord_indices = tf.stack([coord_indices[0],coord_indices[1]], axis=1)
      
      result = tf.tensor_scatter_nd_update(img, r_coord_indices, updates)

      return result

'''





def dummy_median_filter(images):
  
  #images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100
  #images = [ [[[x * n + y + 1] for y in range(n)] for x in range(n)], [[[x * n + y + 1] for y in range(n)] for x in range(n)]   ]

  img = tf.reshape(images, [10,10]  )
  print("image ", img)
  # We generate two outputs as follows:
  # 1. 3x3 patches with stride length 5
  # 2. Same as above, but the rate is increased to 2
  y = tf.image.extract_patches(images=images,
                           sizes=[1, 3, 3, 1],
                           strides=[1, 5, 5, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')

  #print(images)
  print("after image extract patches ", y)
  edges = 3

  #hardcoding the index locations based on the filter size and the image size
  indices = tf.convert_to_tensor( [[1,1],[1,6],[6,1],[6,6]] )
  indices = tf.reshape(indices, [4,2])

  values = tf.math.top_k(y, k=4)[0][:,:,:,-1] # 4 is 50 percentile position out of (3x3=9) elements

  #values = tf.convert_to_tensor(  [ [[13,13],[18,18]],[[63,63],[68,68]] ] )

  #values = tf.random.uniform(shape=[2,1,4], minval=1, maxval=5, dtype=tf.int32)

  values = tf.reshape(values, [-1])
  print("values shape ",values.shape)


  print("after top k values ",values)
  print("after top k indices ",indices)
  result = tf.tensor_scatter_nd_update(img, indices, values)

  print(result)


if __name__ == '__main__':
  
  n = 10
  images = [ [[[x * n + y + 1] for y in range(n)] for x in range(n)]  ]
  print("passing images shape ",np.array(images).shape)
  #sys.exit(0)
  dummy_median_filter(images)

  #indices = tf.convert_to_tensor( [[1,1],[1,6],[6,1],[6,6]] )

  inp = Input(shape=(10,10,1))
  median = median_layer(3,10,10)(inp)
  model_network = Model(inputs=inp, outputs=median)
  model_network.trainable= False #can be applied to a given layer only as well

  images_x = tf.convert_to_tensor(images)
  out_p = model_network(images_x)

  print("got model outp ",out_p)
  

  '''
  import cv2
  image_path = "sample.png"

  sample = cv2.imread(image_path)
  w,h = sample.shape[1], sample.shape[0]
  sample = sample[:,:,0]
  cv2.imshow("loaded sample ",sample)
  cv2.waitKey(0)

  sample = sample/255.0
  sample = sample.reshape((sample.shape[0],sample.shape[1],1))
  print("sample in shape ",sample.shape)
  x = tf.convert_to_tensor(np.array([sample]), dtype=tf.float32)
  print("x shape ",x.shape)


  inp = Input(shape=(128,2048,1))
  median = median_layer(3,128,2048)(inp)
  model_network = Model(inputs=inp, outputs=median)
  model_network.trainable= False #can be applied to a given layer only as well

  sample_out = model_network(x)
  print("sample out shape ",sample_out.shape)
  so = np.array(sample_out)
  print(so)
  cv2.imshow("sample out median filter",so)
  cv2.waitKey(0)
  '''
