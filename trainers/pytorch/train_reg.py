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

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
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


        for i in range(num_batches):
            #self.lr_scheduling(i)


            x_in, y_in = d.get_device_batch() #takes around 0.02s
            #x_in = torch.randn( 16, 8, 200).to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x_in)
            #output.requires_grad = True
            #print("got model input ",x_in)
            #print("got model output ",output)
            #print("got model target ",y_in)
            loss = self.custom_loss(output, y_in)
            #loss = torch.nn.MSELoss()(output, y_in)
            

            #print("grad checks ",output.grad_fn)
            

            #backward and step takes the most time around 1.2 sec
            loss.backward()
            self.optimizer.step()
            

            if i%10==0:
                print("saving model weights ")
                print("loss ",loss.data)
                print("sample target ",y_in[0])
                print("sample model output ",output[0])
                torch.save(self.model.state_dict(), self.weight_save_path )




