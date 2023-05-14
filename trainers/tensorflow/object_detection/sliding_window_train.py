import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
import sys



import numpy as np
import cv2

import copy

import os
import sys
os.environ['top'] = '../../../'
sys.path.append(os.path.join(os.environ['top']))

from dataloaders import imgcsvloader
from applications.lidar_post_detect import params

probe_model_input_output = imgcsvloader.probe_model_input_output

n_kp = params.num_pred_regions


def model_specific_convert_to_tensor(inputs,outputs):
    x = tf.convert_to_tensor(np.array(inputs), dtype=tf.float32)
    
    yc = tf.convert_to_tensor(np.array(outputs[:,:,0]), dtype=tf.float32)
    yr = tf.convert_to_tensor(np.array(outputs[:,:,1:]), dtype=tf.float32)
    
    return x, yc, yr

def train_specific_target_alter(model_output_class_tensor, model_output_reg_tensor, true_label, true_label_class_tensor, true_label_reg_tensor, batch_size = 16, reg_also = True): 
    '''
    use this function to specially alter the model training target from sampled label
    alteration is required for example when instructing the model to ignore loss for predicting something in a multiclass prediction
    for example say model outputs class id ( integer) and bounding box location (float values)
    but say there is nothing there and because there is no object, bounding box location output is useless
    so model should not be penalized for bounding box regression output when object is itself not there
    it should only be penalized for not being able to identify that there was no object (missing classification)
    '''
    target_c = copy.copy(model_output_class_tensor)
    target_r = copy.copy(model_output_reg_tensor)

    #print("model predicted shape",temp.shape)
    #print("target sample ",target[0,0,0,:]) #first channel is batches, second is just an outer layer, third is the number of keypoints, fourth is the types of predictions 

    for b in range(batch_size):#batch size
        for k in range(n_kp): #number of keypoints
            feature_is_present = true_label[b,k,0]

            #reinforce >0 bounding box height and width
            if true_label[b,k,3]==0.0 or true_label[b,k,4]==0.0:
                feature_is_present = 0.0
            
            if feature_is_present==1.0:
                target_c[b,0,k,0] = 1.0
                
                if reg_also: #during initial stages of training do not focus on regression loss
                    target_r[b,0,k,0] = true_label_reg_tensor[b,k,0]
                    target_r[b,0,k,1] = true_label_reg_tensor[b,k,1]
                    target_r[b,0,k,2] = true_label_reg_tensor[b,k,2]
                    target_r[b,0,k,3] = true_label_reg_tensor[b,k,3]
            
            elif feature_is_present==0.0:
                target_c[b,0,k,0] = 0.0
                pass
                #no need to modify the regression target because no need to propagate any loss here
    return target_c, target_r




class trainloop(object):
    def __init__(self, model, dataloader, custom_loss, optimizer=None):
        #for tensorflow backend model needs to be compiled with optimizer
        self.model = model.model
        self.lr = 0.0001

        self.dataloader = dataloader
        self.custom_loss = custom_loss
        self.weight_save_path = params.model_weights_path #"kpmodel"
        
        if not optimizer:
            self.optimizer = Adam(learning_rate=self.lr)
        
        self.model.compile(optimizer=self.optimizer, loss = self.custom_loss)

        

    def fit(self, batches = 20000):
        for b in range(batches):
            _x, _y = self.dataloader.sample_batch()
            x, yc, yr  = model_specific_convert_to_tensor(_x, _y)

            tempc, tempr = self.model.predict(x)
            
            print("tempc shape ",tempc.shape)
            print("tempr shape ",tempr.shape)

            print("yc shape ",yc.shape)
            print("yr shape ",yr.shape)

            reg_also=True
            
            target_c, target_r = train_specific_target_alter(tempc, tempr, _y, yc, yr, self.dataloader.batch_size, reg_also=reg_also)
            

            loss_value = self.model.train_on_batch(x, [target_c, target_r])
            print("loss value ",loss_value)

            yc = tf.reshape(yc, [-1, n_kp, 1])
            combine_actual = tf.keras.layers.concatenate([yc,yr], axis=2)
            combine_predict = tf.keras.layers.concatenate([tempc,tempr], axis=3)


            if b%20==0:
                print("=============== saving model =================")
                self.model.save(self.weight_save_path)

                probe_model_input_output(x,combine_actual, "actual")

                probe_model_input_output(x,combine_predict[0], "model output")

