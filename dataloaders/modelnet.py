#ref:
#https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263

#downloading the model net 10 dataset
#wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
#unzip -q ModelNet10.zip


import numpy as np
import random
import math
from pathlib import Path
import sys
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import sys

path = Path("/datasets/ModelNet10")





def create_pcd_from_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def show_pcd(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    print("Press Q or Excape to exit")
    vis.run()
    vis.destroy_window()

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

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

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)



class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        
        return sampled_points


class dataloader(object):
    def __init__(self, root_dir= path):
        self.batch_size = 2
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.root_dir = root_dir

        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        print("got folders ",folders)
        classes = {folder: i for i, folder in enumerate(folders)}
        self.classes = classes
        print("got classes dict ",classes) #{'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6, 'sofa': 7, 'table': 8, 'toilet': 9}

        #######################################
        #get all the locations of all the pcd files
        self.pcd_file_names = []
        self.pcd_file_label = []

        for category in classes.keys():
            new_dir = root_dir/Path(category)/"train"
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    self.pcd_file_names.append(new_dir/file)
                    self.pcd_file_label.append(classes[category])

        print("total number of pcd training files ",len(self.pcd_file_names)) #3991

        self.datalen = 3991
        self.num_max_points = 1000
        self.name = "modelnet10"
        
        self.prepare_memory_map()

    def sample_one(self, idx=0):
        pcd_file_name = self.pcd_file_names[idx]
        pcd_file_label = self.pcd_file_label[idx]

        '''
        with open(path/"bed/train/bed_0001.off", 'r') as f:
          verts, faces = read_off(f)
        '''
        
        #print("got file name ",pcd_file_name)
        #print("got file label ",pcd_file_label)

        with open(self.root_dir/pcd_file_name, 'r') as f:
          verts, faces = read_off(f)


        pointcloud = PointSampler(self.num_max_points)((verts, faces))
        norm_pointcloud = Normalize()(pointcloud)
        rot_pointcloud = RandRotation_z()(norm_pointcloud)
        noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)

        '''
        print("got pointcloud ",noisy_rot_pointcloud.shape)
        pcd = create_pcd_from_points(noisy_rot_pointcloud)
        show_pcd(pcd)
        '''
        return noisy_rot_pointcloud, pcd_file_label

    def prepare_memory_map(self):
        

        if os.path.exists('x_train'+self.name+'.dat') and os.path.exists('y_train'+self.name+'.dat'):
            print("Data is already memory mapped ")

            self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen, self.num_max_points, 3))
            self.y_train = np.memmap('y_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen, 1)) #labels are single numbers
            return
        

        print("Memory map does not exist, creating one ..")
        self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen, self.num_max_points, 3))
        self.y_train = np.memmap('y_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen, 1))

        idx = 0
        for _ in range(self.datalen):
            points, label = self.sample_one(idx=idx)

            self.x_train[idx,:] = points
            self.y_train[idx,:] = label
            print(idx,end='\r')
            idx+=1
            print(idx,end='\r')

    def get_device_batch(self):

        idx = np.random.randint(self.datalen, size=self.batch_size)

        #normalize, rotate and add noise to each sample in the batch
        x_train = self.x_train[idx]
        #augmntation has already been done in prepare memory map

        x = torch.from_numpy(x_train).view((self.batch_size, self.num_max_points, 3)).to(self.device)
        y = torch.from_numpy(self.y_train[idx]).view((self.batch_size, )).to(self.device).long()

        return x,y






if __name__ == '__main__':
    d = dataloader(path)
    x,y = d.get_device_batch()
    print("dataloader returned train imput shape ",x.size())
    print("dataloader returned train output shape ",y.size())
