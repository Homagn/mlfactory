import numpy as np
import struct
from open3d import *
import open3d as o3d
import glob
import json
from os.path import exists
import cv2
from more_itertools import locate # pip install more-itertools
import math
import time
import random
import pandas as pd
#references to start

# https://towardsdatascience.com/lidar-point-cloud-based-3d-object-detection-implementation-with-colab-part-1-of-2-e3999ea8fdd4
# https://github.com/gkadusumilli/Voxelnet/blob/master/model.py

import os
import sys
import copy

os.environ['top'] = '../../'
sys.path.append(os.path.join(os.environ['top']))

from dataloaders.lidar_datasets import nuscenes
from dataloaders.utils import pointcloud_voxelize

import torch


voxelize = pointcloud_voxelize.voxelize
partition_voxelize = pointcloud_voxelize.partition_voxelize
merge_small_voxels = pointcloud_voxelize.merge_small_voxels




def color_from_labels(points,labels):
    #colorize the pointcloud points based on labels of each point
    background_color = [0.4,0.3,0.3]
    colors = np.array(background_color*points.shape[0]).reshape((points.shape[0],3))

    #all 4 wheelers bluish color
    #road on which the car is driving - black color
    #all static vegetation - greenish color
    #all buildings and static manmade - reddish color
    #everything else - brownish color

    colors[labels==0,:] = [0.2,0.3,0.5]
    colors[labels==1,:] = [0.0,0.0,0.0]
    colors[labels==2,:] = [0.7,0.2,0.1]
    colors[labels==3,:] = [0.1,0.6,0.3]
    return points, colors

def label_from_colors(points, colors):
    labels = np.zeros((points.shape[0],1))
    #print("label from colors points shape ",points.shape)

    color0 = [0.2,0.3,0.5]
    color1 = [0.0,0.0,0.0]
    color2 = [0.7,0.2,0.1]
    color3 = [0.1,0.6,0.3]
    color4 = [0.4,0.3,0.3]


    
    labels[(colors[:,0]==color0[0])&(colors[:,1]==color0[1])&(colors[:,2]==color0[2])] = 0
    labels[(colors[:,0]==color1[0])&(colors[:,1]==color1[1])&(colors[:,2]==color1[2])] = 1
    labels[(colors[:,0]==color2[0])&(colors[:,1]==color2[1])&(colors[:,2]==color2[2])] = 2
    labels[(colors[:,0]==color3[0])&(colors[:,1]==color3[1])&(colors[:,2]==color3[2])] = 3
    labels[(colors[:,0]==color4[0])&(colors[:,1]==color4[1])&(colors[:,2]==color4[2])] = 4

    return points, labels







def probe_segmentation_input_output(model_input, model_output, batch_index = 0, saveloc = ""):
    #model input shape - (batch_size, num_max_points, 3)
    #model output shape - (batch_size, num_classes, num_max_points)
    
    model_input = model_input.cpu().detach().numpy()
    model_output = model_output.cpu().detach().numpy()

    #print("shape of probe model output ",model_output.shape)

    if len(model_output.shape)==3:
        #print("model output has been supplied, taking argmax ")
        model_output = np.argmax(model_output,axis=1)
    if len(model_output.shape)==2:
        #print("model target has been supplied")
        pass
    print("uniques in probe model output ",np.unique(model_output))


    points = model_input[batch_index,:]
    labels = model_output[batch_index,:]



    points, colors = color_from_labels(points,labels)
    pcd = nuscenes.create_pcd_from_points(points,colors)


    '''
    pcd_boxes, voxel_grids = voxelize(points,colors)
    pcd_boxes, voxel_grids = partition_voxelize(points,colors)
    pcd_boxes, voxel_grids = merge_small_voxels(pcd_boxes, voxel_grids)
    '''
    #pcd_boxes, voxel_grids = equally_distribute_points(pcd_boxes, voxel_grids, n_points = 1000)
    #print("sum of pcd_boxes ",sum([np.asarray(i.points).shape[0] for i in pcd_boxes]))
    #also show the voxel grids in visualization
    #pcd_boxes.extend(voxel_grids)

    if saveloc=="":
        #nuscenes.show_pcd([pcd,bb_whole,voxel_grids[10]])
        #nuscenes.show_pcd([bb_whole,voxel_grids[max_count], pcd_boxes[max_count]])
        nuscenes.show_pcd([pcd])

        '''
        for p in range(len(pcd_boxes)):
            nuscenes.show_pcd([pcd_boxes[p], voxel_grids[p]])
        '''

        '''
        for i in range(20):
            nuscenes.show_pcd([pcd_boxes[i]]) #examine the points in the first pcd box
        '''

    else:
        o3d.io.write_point_cloud(saveloc, pcd)










def train_chunk_acceptable(pcd):
    n_pts = np.asarray(pcd.points).shape[0]
    u,c = np.unique(np.asarray(pcd.colors), axis=0, return_counts=True)
    car_color_index = np.where(u==0.5) #blue channel color value of pointcloud belonging to car is made =0.5
    #print("car color index ",car_color_index)
    if car_color_index[0].shape[0]==0:
        num_car_points = 0
    else:
        num_car_points = c[car_color_index[0][0]]
    
    if n_pts>1000 and u.shape[0]>=4 and num_car_points>50: #atleast 50 points for each unique class and at least 4 unique classes
        return True
    else:
        return False








def random_input_chunk(model_input, model_output, pointcloud_scaling = 1000.0, visualize = True):
    #taking 0 for batch size has been set to always use 1 in the dataloader class
    #take note of variable random_part where the random chunk is taken out of the entire pcd data
    points = model_input[0,:] 

    
    points = points * pointcloud_scaling



    labels = model_output[0,:]

    #print("check if array contains NAN ", pd.isna(points).any(), pd.isna(labels).any())
    #print("got input chunks points shape ",points.shape)
    #print("got input chunks labels shape ",labels.shape)


    points, colors = color_from_labels(points,labels)

    '''
    ############################
    #show the original entire pcd with ground truth coloring
    print("random input chunks function, showing original entire pcd ")
    pcd = nuscenes.create_pcd_from_points(points,colors)
    nuscenes.show_pcd([pcd])
    ############################
    '''


    pcd_boxes, voxel_grids = voxelize(points,colors)
    pcd_boxes, voxel_grids = partition_voxelize(points,colors)
    pcd_boxes, voxel_grids = merge_small_voxels(pcd_boxes, voxel_grids)


    '''
    ############################
    #get the number of points inside each voxel partition
    for pcd in pcd_boxes:
        print("size ",np.asarray(pcd.points).shape[0])
    ############################
    '''

    '''
    ############################
    #show the pcd voxel along with their bounding boxes and points
    print("showing merged voxel grids ")
    pcd_boxes.extend(voxel_grids)
    nuscenes.show_pcd(pcd_boxes)
    ############################
    '''



    #some pcd_parts may contain very less points so take care of that
    acceptable_idcs = []
    for idx in range(len(pcd_boxes)):

        '''
        n_pts = np.asarray(pcd_boxes[idx].points).shape[0]
        u,c = np.unique(np.asarray(pcd_boxes[idx].colors), axis=0, return_counts=True)
        car_color_index = np.where(u==0.5) #blue channel color value of pointcloud belonging to car is made =0.5
        print("car color index ",car_color_index)
        if car_color_index[0].shape[0]==0:
            num_car_points = 0
        else:
            num_car_points = c[car_color_index[0][0]]
        
        if n_pts>1000 and u.shape[0]>=4 and num_car_points>50: #atleast 50 points for each unique class and at least 4 unique classes
            acceptable_idcs.append(idx)
        '''

        if train_chunk_acceptable(pcd_boxes[idx]):
            acceptable_idcs.append(idx)


    if acceptable_idcs==[]: #could not match criteria
        return np.array([]), None, None

    #random_part = np.random.randint(len(pcd_boxes))
    random_part = np.random.choice(acceptable_idcs)
    sample = pcd_boxes[random_part]

    sample_pts, sample_cls = np.asarray(pcd_boxes[random_part].points), np.asarray(pcd_boxes[random_part].colors)
    points, labels = label_from_colors(sample_pts, sample_cls)

    labels = labels.reshape((labels.shape[0],))
    

    #normalization happens here
    points = augment(points)

    if visualize:
        print("after chunking ")
        print("got input chunks points shape ",points.shape)
        print("got input chunks labels shape ",labels.shape)

        points, colors = color_from_labels(points,labels)
        pcd = nuscenes.create_pcd_from_points(points,colors)
        nuscenes.show_pcd([pcd])

    return points, labels, labels.shape[0]













class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        #norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        
        #move the point cloud in the quadrant where every value is positive

        norm_pointcloud = pointcloud -  np.min(pointcloud)



        norm_pointcloud = norm_pointcloud/ np.max(np.linalg.norm(norm_pointcloud, axis=1))

        #print("sample norm pointcloud ",norm_pointcloud)

        return  norm_pointcloud

class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud


def augment(pointcloud):
    norm_pointcloud = Normalize()(pointcloud)
    #rot_pointcloud = RandRotation_z()(norm_pointcloud)
    #noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)
    return norm_pointcloud



class dataloader(object):
    def __init__(self, dataset = "nuscenes"):
        self.batch_size = 1 #always using 1 for now
        self.dataset = dataset
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.num_max_points = 50000 #maximum number of points in the pointcloud (set it arbitrarily high)

        if self.dataset=="nuscenes":
            self.num_lidar_files = 2500 #actually 2700 in part1 folder but using 100 for faster testing
            self.name = 'nuscenes'
        

        self.prepare_memory_map()

    def init_nuscenes(self):
        self.dataset_name = '/datasets/nuscenes/'
        self.data_part = 1

        #label_files_json = dataset_name+'nuScenes-lidarseg-all-v1.0/v1.0-trainval/lidarseg.json'
        self.label_files_json = self.dataset_name+'nuScenes-panoptic-v1.0-all/v1.0-trainval/panoptic.json'
        self.meta_files_json = self.dataset_name+'v1.0-trainval_meta/v1.0-trainval/sample_data.json'
        self.calibrated_sensor_json = self.dataset_name+'v1.0-trainval_meta/v1.0-trainval/calibrated_sensor.json'
        self.ego_pose_json = self.dataset_name+'v1.0-trainval_meta/v1.0-trainval/ego_pose.json'
        #labels_folder = dataset_name+'nuScenes-lidarseg-all-v1.0/'
        self.labels_folder = self.dataset_name+'nuScenes-panoptic-v1.0-all/'

        self.label_data, self.meta_data, self.calib_sensor_data, self.ego_pose_data = nuscenes.load_meta_files(self.label_files_json, self.meta_files_json, self.calibrated_sensor_json, self.ego_pose_json)

    def sample_nuscenes(self, idx):
        pcd_location, pcd_seg_location = nuscenes.get_lidar_files_nth(self.label_data, self.meta_data, self.calib_sensor_data, self.ego_pose_data, self.labels_folder, self.data_part, idx)

        bin_file = pcd_location
        label_file = pcd_seg_location
        class_ids = [15,16,17,18,19,20,21,22,23,  24, 28, 30] #15 to 23 are all 4 wheelers


        pcd_parts = nuscenes.get_pcd_parts(bin_file, label_file, class_ids)

        #for visualization only 

        #pcd_ = pcd_parts[0]
        #nuscenes.show_pcd([pcd_])

        all_points = np.array([])
        all_labels = np.array([])

        for k in pcd_parts.keys():
            #if k==0 or k==-1:
            if k==0:
                continue #this is the entire pointcloud stored for quick lookup
            for pcd in pcd_parts[k]:
                p = np.asarray(pcd.points)
                
                

                if k>=15 and k<=23: #all 4 wheelers
                    #assign a label of 0
                    l = np.zeros((p.shape[0],1))
                elif k==24:
                    #assign label 1
                    l = np.ones((p.shape[0],1))
                elif k==28:
                    #assign label 2
                    l = 2*np.ones((p.shape[0],1))
                elif k==30:
                    #assign label 3
                    l = 3*np.ones((p.shape[0],1))
                else: #all other class categories
                    #assign label 4
                    l = 4*np.ones((p.shape[0],1))



                if all_points.shape[0]==0:
                    all_points = p
                    all_labels = l
                else:
                    all_points = np.concatenate((all_points,p), axis=0)
                    all_labels = np.concatenate((all_labels,l), axis=0)

        #print("all_points shape ",all_points.shape)
        #print("all labels shape ",all_labels.shape)

        '''
        rem = self.num_max_points - all_points.shape[0]
        #rem = np.asarray(pcd_parts[0].points).shape[0] - all_points.shape[0]

        pcd_rem = pcd_parts[-1][0]
        #print("appending remaining points ")
        pcd_rem_points = np.asarray(pcd_rem.points)[:rem,:] #take the first rem number of points
        pcd_rem_labels = 4*np.ones((pcd_rem_points.shape[0],1))
        all_points = np.concatenate((all_points,pcd_rem_points), axis=0)
        all_labels = np.concatenate((all_labels,pcd_rem_labels), axis=0)

        
        if all_points.shape[0]!=self.num_max_points:
            #this can happen when there are not enough points in the entire lidar scene recording
            return np.array([]), np.array([])
        '''
        



        return all_points, all_labels


    def prepare_memory_map(self):
        

        if os.path.exists('/datasets/x_train'+self.name+'.dat') and os.path.exists('/datasets/y_train'+self.name+'.dat'):
            print("Data is already memory mapped ")

            self.x_train = np.memmap('/datasets/x_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.num_lidar_files, self.num_max_points, 3))
            self.y_train = np.memmap('/datasets/y_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.num_lidar_files, self.num_max_points))
            self.train_meta = np.memmap('/datasets/meta'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.num_lidar_files, 1))
            return
        
        if self.dataset=="nuscenes":
            self.init_nuscenes()

        print("Memory map does not exist, creating one ..")
        self.x_train = np.memmap('/datasets/x_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.num_lidar_files, self.num_max_points, 3))
        self.y_train = np.memmap('/datasets/y_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.num_lidar_files, self.num_max_points))
        self.train_meta = np.memmap('/datasets/meta'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.num_lidar_files, 1))

        idx = 0
        for _ in range(self.num_lidar_files):
            points, labels = self.sample_one(idx=idx)

            '''
            if points.shape[0]==0:
                continue
            else:
            '''

            #labels = labels.reshape((labels.shape[0],))
            #self.x_train[idx,:] = points
            #self.y_train[idx,:] = labels

            labels = labels.reshape((labels.shape[0],))
            self.x_train[idx,:labels.shape[0]] = points
            self.y_train[idx,:labels.shape[0]] = labels
            self.train_meta[idx] = labels.shape[0]

            print(idx,end='\r')
            idx+=1



        if self.num_lidar_files!=idx:
            print("change self num lidar files to ",idx)
            self.num_lidar_files = idx





    def sample_one(self,idx=0):
        if self.dataset=="nuscenes":
            points, labels = self.sample_nuscenes(idx)

            return points,labels


    def get_device_batch(self):

        idx = np.random.randint(self.num_lidar_files, size=self.batch_size)

        #normalize, rotate and add noise to each sample in the batch
        
        #x_train = self.x_train[idx] #consider this to be the direct output of a lidar with self.num_max_points
        #y_train = self.y_train[idx]


        train_meta = int(self.train_meta[idx][0][0])
        x_train = self.x_train[idx][0][:train_meta,:] #consider this to be the direct output of a lidar with self.num_max_points
        y_train = self.y_train[idx][0][:train_meta]

        x_train = x_train.reshape((self.batch_size,x_train.shape[0],3))
        y_train = y_train.reshape((self.batch_size,y_train.shape[0]))
        
        #print("In get device batch ")
        #print("idx ",idx," number of points ",train_meta," x_train shape ",x_train.shape, " y_train shape ",y_train.shape)



        #pointcloud typically consists of 30,000+ points which is too many points for pointnet
        #so take a random chunk of closely connected points (voxel) typically containing around 1500 points (number not fixed, thats why batch size 1 is being used)
        #also locally normalize the selected points about the center of the bounding box
        x_train, y_train, num_points = random_input_chunk(x_train, y_train, visualize = False)


        if x_train.shape[0]!=0:
            x = torch.from_numpy(x_train).view((self.batch_size, num_points, 3)).to(self.device, dtype=torch.float)
            y = torch.from_numpy(y_train).view((self.batch_size, num_points)).to(self.device).long()

            return x,y

        else:
            #try again
            print("WARNING could not random chunk points according to criteria, trying again")
            return self.get_device_batch()



        







if __name__ == '__main__':

    d = dataloader(dataset="nuscenes")
    i,o = d.get_device_batch()
    print("input shape output shape ",i.size(), o.size())

    probe_segmentation_input_output(i,o, batch_index=0)

