#ref - https://keras.io/examples/vision/pointnet_segmentation/
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import torch
import open3d as o3d

def download_init_shapenet():
    dataset_url = "https://git.io/JiY4i"

    print("getting dataset file using keras utils")
    dataset_path = keras.utils.get_file(
        fname="shapenet.zip",
        origin=dataset_url,
        cache_subdir="datasets",
        hash_algorithm="auto",
        extract=True,
        archive_format="auto",
        cache_dir="datasets",
    )


    print("opening json file ...")
    with open("/tmp/.keras/datasets/PartAnnotation/metadata.json") as json_file:
        metadata = json.load(json_file)

    print(metadata)


    print("extracting shapenet data corresponding to Airplane category ")

    points_dir = "/tmp/.keras/datasets/PartAnnotation/{}/points".format(
        metadata["Airplane"]["directory"]
    )
    labels_dir = "/tmp/.keras/datasets/PartAnnotation/{}/points_label".format(
        metadata["Airplane"]["directory"]
    )
    LABELS = metadata["Airplane"]["lables"]
    COLORS = metadata["Airplane"]["colors"]

    VAL_SPLIT = 0.2
    NUM_SAMPLE_POINTS = 1024
    BATCH_SIZE = 32
    EPOCHS = 60
    INITIAL_LR = 1e-3




    point_clouds, test_point_clouds = [], []
    point_cloud_labels, all_labels = [], []

    points_files = glob(os.path.join(points_dir, "*.pts"))
    for point_file in tqdm(points_files):
        point_cloud = np.loadtxt(point_file)
        if point_cloud.shape[0] < NUM_SAMPLE_POINTS:
            continue

        # Get the file-id of the current point cloud for parsing its
        # labels.
        file_id = point_file.split("/")[-1].split(".")[0]
        label_data, num_labels = {}, 0
        for label in LABELS:
            label_file = os.path.join(labels_dir, label, file_id + ".seg")
            if os.path.exists(label_file):
                label_data[label] = np.loadtxt(label_file).astype("float32")
                num_labels = len(label_data[label])

        # Point clouds having labels will be our training samples.
        try:
            label_map = ["none"] * num_labels
            for label in LABELS:
                for i, data in enumerate(label_data[label]):
                    label_map[i] = label if data == 1 else label_map[i]
            label_data = [
                LABELS.index(label) if label != "none" else len(LABELS)
                for label in label_map
            ]
            # Apply one-hot encoding to the dense label representation.
            label_data = keras.utils.to_categorical(label_data, num_classes=len(LABELS) + 1)

            point_clouds.append(point_cloud)
            point_cloud_labels.append(label_data)
            all_labels.append(label_map)
        except KeyError:
            test_point_clouds.append(point_cloud)



    print("number of pointclouds and labels ",len(point_clouds), len(all_labels))
    print("type of point_clouds and labels  ",type(point_clouds[0]), type(all_labels[0]) )
    print("pointcloud labels type and shape ",type(point_cloud_labels), point_cloud_labels[0].shape)
    print("sample point cloud labels ",point_cloud_labels[0])


    return point_clouds, point_cloud_labels


def create_pcd_from_points(points,colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def color_from_labels(points,labels):
    #colorize the pointcloud points based on labels of each point
    background_color = [0.4,0.3,0.3]
    colors = np.array(background_color*points.shape[0]).reshape((points.shape[0],3))

    colors[labels==0,:] = [0.2,0.3,0.5]
    colors[labels==1,:] = [0.0,0.0,0.0]
    colors[labels==2,:] = [0.7,0.2,0.1]
    colors[labels==3,:] = [0.1,0.6,0.3]
    return points, colors

def show_pcd(pcds):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for pcd in pcds:
        vis.add_geometry(pcd)
    # run visualizer main loop
    print("Press Q or Excape to exit")
    vis.run()
    vis.destroy_window()

def probe_segmentation_input_output(model_input, model_output, batch_index = 0, saveloc = ""):

    
    model_input = model_input.cpu().detach().numpy()
    model_output = model_output.cpu().detach().numpy()

    #print("shape of probe model output ",model_output.shape)

    if len(model_output.shape)==3:
        #print("model output has been supplied, taking argmax ")
        model_output = np.argmax(model_output,axis=1)
    if len(model_output.shape)==2:
        #print("model target has been supplied")
        pass
    #print("uniques in probe model output ",np.unique(model_output))


    points = model_input[batch_index,:]
    labels = model_output[batch_index,:]



    points, colors = color_from_labels(points,labels)
    pcd = create_pcd_from_points(points,colors)


    if saveloc=="":
        show_pcd([pcd])

    else:
        o3d.io.write_point_cloud(saveloc, pcd)



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

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)


def augment(pointcloud):
    norm_pointcloud = Normalize()(pointcloud)
    #rot_pointcloud = RandRotation_z()(norm_pointcloud)
    #noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)
    return norm_pointcloud





class dataloader(object):
    def __init__(self):
        self.num_lidar_files = 3694
        self.num_max_points = 3000
        self.name = "shapenet"
        self.batch_size = 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.prepare_memory_map()

    def sample_one(self, idx=0):
        #used in tensorflow
        categorical_labels = self.point_cloud_labels[idx]
        #used in pytorch
        index_labels = np.argmax(categorical_labels, axis=1)
        return self.point_clouds[idx], index_labels

    def prepare_memory_map(self):
        name = self.name
        num_lidar_files = self.num_lidar_files
        num_max_points = self.num_max_points

        if os.path.exists('/datasets/x_train'+name+'.dat') and os.path.exists('/datasets/y_train'+name+'.dat'):
            print("Data is already memory mapped ")

            self.x_train = np.memmap('/datasets/x_train'+name+'.dat', dtype=np.float32, mode='r', shape=(num_lidar_files, num_max_points, 3))
            self.y_train = np.memmap('/datasets/y_train'+name+'.dat', dtype=np.float32, mode='r', shape=(num_lidar_files, num_max_points))
            #store the number of points each lidar pointcloud has
            self.train_meta = np.memmap('/datasets/meta'+name+'.dat', dtype=np.float32, mode='r', shape=(num_lidar_files, 1))
            return
        

        self.point_clouds, self.point_cloud_labels = download_init_shapenet()

        print("Memory map does not exist, creating one ..")
        self.x_train = np.memmap('/datasets/x_train'+name+'.dat', dtype=np.float32, mode='w+', shape=(num_lidar_files, num_max_points, 3))
        self.y_train = np.memmap('/datasets/y_train'+name+'.dat', dtype=np.float32, mode='w+', shape=(num_lidar_files, num_max_points))
        self.train_meta = np.memmap('/datasets/meta'+name+'.dat', dtype=np.float32, mode='w+', shape=(num_lidar_files, 1))


        idx = 0
        for _ in range(num_lidar_files):
            points, labels = sample_one(idx=idx)


            labels = labels.reshape((labels.shape[0],))
            self.x_train[idx,:labels.shape[0]] = points
            self.y_train[idx,:labels.shape[0]] = labels
            self.train_meta[idx] = labels.shape[0]

            print(idx,end='\r')
            idx+=1

    def get_device_batch(self):
        #for now batch size is always 1
        batch_size = 1
        idx = np.random.randint(self.num_lidar_files, size=batch_size)

        train_meta = int(self.train_meta[idx][0][0])

        x_train = self.x_train[idx][0][:train_meta,:] #consider this to be the direct output of a lidar with self.num_max_points
        y_train = self.y_train[idx][0][:train_meta]

        #print("x_train y_train shape and train_meta ",x_train.shape, y_train.shape, train_meta)

        x_train = augment(x_train)
        #print("after augmentation")
        #print("max,min of x_train ",np.max(x_train), np.min(x_train)) #verified to be in 0-1 range

        
        x = torch.from_numpy(x_train).view((batch_size, train_meta, 3)).to(self.device, dtype=torch.float)
        y = torch.from_numpy(y_train).view((batch_size, train_meta)).to(self.device).long()
        return x,y


if __name__ == '__main__':
    d = dataloader()
    i,o = d.get_device_batch()
    print("input shape output shape ",i.size(), o.size())

    probe_segmentation_input_output(i,o, batch_index=0)





