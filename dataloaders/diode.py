import os
import sys

import numpy as np
import cv2

import torch

os.environ['top'] = '../'
sys.path.append(os.path.join(os.environ['top']))
from visualizers import pointcloud
from scipy import ndimage

def read_files():
    path = "/datasets/diode_train/indoors"

    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    filelist.sort()
    data = {
        "image": [x for x in filelist if x.endswith(".png")],
        "depth": [x for x in filelist if x.endswith("_depth.npy")],
        "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
    }

    print("sample path names ..................")
    print("image ",data["image"][0])
    print("depth ",data["depth"][0])
    print("mask ",data["mask"][0])
    return data, len(data["image"])

#https://github.com/diode-dataset/diode-devkit/issues/3
#diode reports issues with super sharp edges in depth maps which is wrong
#https://github.com/diode-dataset/diode-devkit/issues/3
#so use the edges function to create a clean pointcloud


def edges(d):
    #single derivative kind of does the job, adding double derivative does even better job

    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dx2 = ndimage.sobel(dx, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    dy2 = ndimage.sobel(dy, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)+ np.abs(dx2) + np.abs(dy2)


'''
def edges(d):
    dx = ndimage.prewitt(d, 0)  # horizontal derivative
    dy = ndimage.prewitt(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)
'''

def load(image_path, dm, validity_mask):
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    validity_mask = np.load(validity_mask)
    dm= np.load(dm).squeeze()


    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.5
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    dm = np.log(dm, where=validity_mask)

    dm = np.ma.masked_where(~validity_mask, dm)
    dm[edges(dm) > 0.5] = 0.0  # Hide depth edges   

    #cmap = plt.cm.jet
    #cmap.set_bad(color='black')
    #plt.imshow(dm, cmap=cmap, vmax=np.log(MAX_DEPTH))

    return image, dm, np.max(dm), validity_mask






class dataloader(object):
    def __init__(self):
        self.batch_size = 8
        self.w = 256
        self.h = 192

        #sometimes model might output a different size image
        self.w_target = 160
        self.h_target = 120
        self.nc = 3

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.additional_scaling = 3.0 #sometimes max depth in depth map can be 3

        self.data, self.datalen = read_files()
        print("Number of examples in the dataset ",self.datalen)
        self.data_fraction = 1
        self.datalen = self.datalen//self.data_fraction
        self.name = 'diode'

        self.prepare_memory_map()

    def prepare_memory_map(self): #should only take care of image resize and normalization/ all other image transforms in load
        if os.path.exists('x_train'+self.name+'.dat') and os.path.exists('y_train'+self.name+'.dat'):
            print("Data is already memory mapped ")
            if self.nc>1:
                self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen, self.h, self.w, self.nc))
            else:
                self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen, self.h, self.w))
            self.y_train = np.memmap('y_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen, self.h_target, self.w_target))
            return

        print("Memory map does not exist, creating one ..")
        if self.nc>1:
            self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen, self.h, self.w, self.nc))
        else:
            self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen, self.h, self.w))
        self.y_train = np.memmap('y_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen, self.h_target, self.w_target))

        #max_y = self.additional_scaling
        #actually the max will be around 3.0, which is wrong for if you visualize the dataset
        #I think generally in nyuv2 its 65000, so lets make it that range
        #entire range if squashed between 0 and 1 the loss propagated will be very low and model wont train properly
        #models for regression generally train well with huge ranges and high resolution
        max_y = 1.0/22000.0

        for idx in range(self.datalen):
            print(idx,end='\r')

            x, y = self.sample_one(sample_number=idx)

            x = cv2.resize(x,(self.w,self.h))
            if self.nc==1:
                #change to grayscale because depth estimation should not be biased on color but just the overall intensity
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

            x = np.array(x/255.0,dtype = np.float32)

            y = cv2.resize(y,(self.w_target,self.h_target))
            y = np.array(y/max_y,dtype = np.float32) #lets try not normalizing the depth map and put relu in the last layer via loss


            self.x_train[idx,:] = x
            self.y_train[idx,:] = y

            #print("min and max of x ",np.min(x), np.max(x))
            #print("min and max of y ",np.min(y), np.max(y))


    def sample_one(self, sample_number = -1):
        #print("number of files in dataset ",self.datalen)
        if sample_number==-1:
            sample_number = np.random.randint(0, self.datalen)
        #print("viewing sample number ",sample_number)
        rgb, d, md, mask = load(self.data["image"][sample_number], self.data["depth"][sample_number], self.data["mask"][sample_number])
        return rgb, d

    def get_device_batch(self):
        idx = np.random.randint(self.datalen, size=self.batch_size)

        x = torch.from_numpy(self.x_train[idx]).view((self.batch_size, 1, self.h, self.w, self.nc)).to(self.device)
        y = torch.from_numpy(self.y_train[idx]).view((self.batch_size, 1, self.h_target, self.w_target)).to(self.device)

        return x,y




if __name__ == '__main__':

    dt = dataloader()
    rgb, d = dt.sample_one()


    print("got maximum of depth map from sampled map ",np.max(d))

    #diode dataset
    #from - https://github.com/diode-dataset/diode-devkit/blob/master/intrinsics.txt
    #multiply the value found in fx_d and fy_d by 2
    diode_camera_params = {
                        "fx": 2*886.81,
                        "fy": 2*927.06,
                        "centerX": 512.0,
                        "centerY": 384.0,
                        "scalingFactor": 20000
                        }

    pointcloud.show_pcd_from_rgbd(rgb, 255.0*d/np.max(d), diode_camera_params)


    xt, yt = dt.get_device_batch()
    print("sample tensor batch ",type(xt), xt.shape)


