import torch
from torch.utils import data
from torch import nn as nn
from torchvision import models

import torch.nn.functional as F
import cv2

import numpy as np



class trainloop(object):
    def __init__(self, model, dataloader, custom_loss, optimizer = None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.dataloader = dataloader
        self.custom_loss = custom_loss
        self.weight_save_path = "depthpred_weights.pth"
        if not optimizer:
            #self.optimizerlr = 0.0001
            #self.optimizer = torch.optim.Adam(self.model.parameters(),weight_decay=1e-5,lr=self.optimizerlr)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
        else:
            self.optimizer = optimizer
        self.model.to(self.device)

    def load_prev_weights(self):
        print("Trying to load model weights ")
        self.model.load_state_dict(torch.load(self.weight_save_path))
        print("loaded weights")

    def lr_scheduling(self, batch_number):
        if batch_number>10000 and batch_number%1000==0:
            print("learning rate changed ")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.99


    def fit(self, num_batches = 800000, verbose = True, save_predictions = True):
        save_predictions = True
        d = self.dataloader
        batch_size = d.batch_size
        w = d.w
        h = d.h

        for i in range(num_batches):
            self.lr_scheduling(i)

            im1,im2 = d.get_device_batch() #takes around 0.02s
            self.optimizer.zero_grad()

            output = self.model(im1)

            #print("mean output ",torch.mean(output))
            #print("max output ",torch.max(output))

            loss = self.custom_loss(output, im2)
            #backward and step takes the most time around 1.2 sec
            loss.backward()
            self.optimizer.step()
            print("loss ",loss.data)

            if i%10==0:
                print("saving model weights ")
                torch.save(self.model.state_dict(), self.weight_save_path )

                if save_predictions:
                    #seg_pred = F.sigmoid(output).detach().cpu().numpy().reshape((batch_size,w,h))
                    seg_pred = self.model.inference_additional_ops(output).reshape((batch_size,d.h_target,d.w_target))

                    target = im2.cpu().numpy().reshape((batch_size,d.h_target,d.w_target))
                    
                    inputs = im1.cpu().numpy().reshape((batch_size,h,w,3))

                    #for binary segmentation I think the hot colormap will give just yellow and black
                    save_pred = cv2.applyColorMap(np.array(seg_pred[0],dtype = np.uint8) , cv2.COLORMAP_HSV)
                    save_target = cv2.applyColorMap(np.array(target[0],dtype = np.uint8) , cv2.COLORMAP_HSV)

                    cv2.imwrite("predictions/"+repr(i%300)+'.png',save_pred)
                    #cv2.imwrite("predictions/"+repr(i%100)+'.png',seg_pred[0]*100.0)
                    cv2.imwrite("predictions/target_"+repr(i%300)+'.png',save_target)
                    
                    cv2.imwrite("predictions/input_"+repr(i%300)+'.png',inputs[0]*255.0)




