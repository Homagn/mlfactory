import os
import sys

import numpy as np
import cv2

import torch
from glob import glob



def read_files():
    path = "/datasets/CIHP/human_part_seg/Training"
    NUM_TRAIN_IMAGES = 30000
    train_images = sorted(glob(os.path.join(path, "Images/*")))[:NUM_TRAIN_IMAGES]
    train_masks = sorted(glob(os.path.join(path, "Category_ids/*")))[:NUM_TRAIN_IMAGES]

    data = {"image":train_images, "masks":train_masks}

    return data, len(data["image"])


def load(image_path, mask_path):
    x = cv2.imread(image_path)
    y = cv2.imread(mask_path,0) #force read grayscale
    y[y>0] =1

    return x, y





class dataloader(object):
    def __init__(self):
        self.batch_size = 8
        self.w = 256
        self.h = 256
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.additional_scaling = 1.0 #CIHP dataset human segmentation does not have any additional scaling

        self.data, self.datalen = read_files()
        print("Number of examples in the dataset ",self.datalen)
        self.data_fraction = 1
        self.name = 'cihp'

        self.prepare_memory_map()

    def prepare_memory_map(self): #should only take care of image resize and normalization/ all other image transforms in load
        if os.path.exists('x_train'+self.name+'.dat') and os.path.exists('y_train'+self.name+'.dat'):
            print("Data is already memory mapped ")
            self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen, 256, 256, 3))
            self.y_train = np.memmap('y_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen, 256, 256))
            return

        print("Memory map does not exist, creating one ..")
        self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen, 256, 256, 3))
        self.y_train = np.memmap('y_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen, 256, 256))

        max_y = self.additional_scaling

        for idx in range(self.datalen//self.data_fraction):
            print(idx,end='\r')

            x, y = self.sample_one(sample_number=idx)

            x = cv2.resize(x,(self.w,self.h))
            x = np.array(x/255.0,dtype = np.float32)

            y = cv2.resize(y,(self.w,self.h))
            y = np.array(y/max_y,dtype = np.float32)


            self.x_train[idx,:] = x
            self.y_train[idx,:] = y

            #print("min and max of x ",np.min(x), np.max(x))
            #print("min and max of y ",np.min(y), np.max(y))


    def sample_one(self, sample_number = -1):
        #print("number of files in dataset ",self.datalen)
        if sample_number==-1:
            sample_number = np.random.randint(0, self.datalen)
        #print("viewing sample number ",sample_number)
        rgb, mask= load(self.data["image"][sample_number], self.data["masks"][sample_number])
        return rgb, mask

    def get_device_batch(self):
        idx = np.random.randint(self.datalen, size=self.batch_size)

        x = torch.from_numpy(self.x_train[idx]).view((self.batch_size, 3, self.w, self.h)).to(self.device)
        y = torch.from_numpy(self.y_train[idx]).view((self.batch_size, 1, self.w, self.h)).to(self.device)

        return x,y




if __name__ == '__main__':

    dt = dataloader()
    rgb, d = dt.sample_one()



    cv2.imshow("rgb",rgb)
    cv2.waitKey(0)

    cv2.imshow("depth ", d/np.max(d))
    cv2.waitKey(0)

    print("max of depth map ",np.max(d))
    print("min of depth map ",np.min(d))


    xt, yt = dt.get_device_batch()
    print("sample tensor batch ",type(xt), xt.shape)


