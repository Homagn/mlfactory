import torch
from torch.utils import data
from torch import nn as nn
from torchvision import models

import torch.nn.functional as F
import cv2

import numpy as np



class trainloop(object):
    def __init__(self, model, dataloader, custom_loss, optimizer = None, viz_pred_func = None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.dataloader = dataloader
        self.custom_loss = custom_loss
        self.weight_save_path = "pointseg_weights.pth"
        self.viz_pred_func = viz_pred_func
        if not optimizer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
        else:
            self.optimizer = optimizer
        self.model.to(self.device)
        #for training pointclouds using gradient accumulation approach
        #basically using dataloader batch size always 1, but do the trick in training by averaging loss
        #https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
        self.accum_iter = 16 #effective batch size of 16

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

        self.loss_hist = []
        self.num_pred_saves = 20
        #very high weights to 4 wheeler class and low weight to unidentified class
        self.custom_class_weighting = torch.tensor([2,1,1,1,0.5]).to(d.device)

        for i in range(num_batches):
            #self.lr_scheduling(i)

            pcloud,labels = d.get_device_batch() #takes around 0.02s

            #to do
            #pointcloud input selection based on density of cloud as well
            #weighted nll loss function
            #correct the prepare memory map function, where label 4 is being assigned wrongly


            '''
            #print("trainloop pcloud shape ",pcloud.size())
            self.optimizer.zero_grad()

            outputs, m3x3, m64x64 = self.model(pcloud.transpose(1,2))

            #print("trainloop outputs shape ",outputs.size(), outputs) #(batch_size, 10)
            #print("trainloop labels shape ",labels.size(), labels)# (batch_size, )

            loss = self.custom_loss(outputs, labels, m3x3, m64x64)
            loss.backward()
            self.optimizer.step()

            print("loss ",loss.data)
            '''




            # passes and weights update
            with torch.set_grad_enabled(True):
                # forward pass 
                #preds = model(inputs)
                preds, m3x3, m64x64 = self.model(pcloud.transpose(1,2))
                #loss  = criterion(preds, labels)
                loss = self.custom_loss(preds, labels, m3x3, m64x64, self.custom_class_weighting)

                # normalize loss to account for batch accumulation
                loss = loss / self.accum_iter 

                # backward pass
                loss.backward()

                # weights update
                if ((i + 1) % self.accum_iter == 0):
                    self.optimizer.step()
                    self.optimizer.zero_grad()


                    #record and store the loss, also save the best predictions
                    print("sample loss ",loss.data*self.accum_iter)
                    self.loss_hist.append(loss.data*self.accum_iter)
                    self.loss_hist.sort()
                    if len(self.loss_hist)>self.num_pred_saves:
                        self.loss_hist.pop(-1) #pop the visualization corresponsing to the lossiest one

                    if save_predictions:
                        if self.viz_pred_func==None:
                            pass
                        else:
                            try:
                                position = self.loss_hist.index(loss.data*self.accum_iter)
                                saveloc = "predictions/"+repr(position)+'.pcd'
                                #probe_segmentation_input_output(pcloud, outputs, batch_index = 0, saveloc = saveloc)
                                self.viz_pred_func(pcloud, preds, batch_index = 0, saveloc = saveloc)
                            except:
                                print("not saving prediction this time as could not find index")
                                pass
                            








            
            if i%64==0:
                print("saving model weights ")
                torch.save(self.model.state_dict(), self.weight_save_path )

                





