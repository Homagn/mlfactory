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



def create_pcd_from_points(points,colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def voxelize(points, colors, point_density = 1000, pcd_boxes = [], voxel_grids = []):
    voxel_vol = points.shape[0]//1000
    res = math.ceil(math.pow(voxel_vol,(1/3)))
    

    pcd = create_pcd_from_points(points,colors)

    bb_whole = o3d.geometry.OrientedBoundingBox.get_axis_aligned_bounding_box(pcd)
    box_corners = np.asarray(bb_whole.get_box_points())
    #print("got bounding box for the whole pcd ",box_corners)

    x_bounds = [math.floor(np.min(box_corners[:,0])), math.ceil(np.max(box_corners[:,0]))]
    y_bounds = [math.floor(np.min(box_corners[:,1])), math.ceil(np.max(box_corners[:,1]))]
    z_bounds = [math.floor(np.min(box_corners[:,2])), math.ceil(np.max(box_corners[:,2]))]

    x_res,y_res,z_res = math.ceil((x_bounds[1]-x_bounds[0])/res), math.ceil((y_bounds[1]-y_bounds[0])/res), math.ceil((z_bounds[1]-z_bounds[0])/res)
    #print("box size x,y,z/ resolution step ",x_res, y_res, z_res, res)


    count=0
    max_count = 0
    max_points = 0


    x_i = list(range(x_bounds[0],x_bounds[1]+x_res, x_res))
    y_i = list(range(y_bounds[0],y_bounds[1]+y_res, y_res))
    z_i = list(range(z_bounds[0],z_bounds[1]+z_res, z_res))

    # x_i = list(np.linspace(x_bounds[0],x_res,x_bounds[1]+math.ceil(x_res)))
    # y_i = list(np.linspace(y_bounds[0],y_res,y_bounds[1]+math.ceil(y_res)))
    # z_i = list(np.linspace(z_bounds[0],z_res,z_bounds[1]+math.ceil(z_res)))

    p_box_accum = np.empty((0,3))
    c_box_accum = np.empty((0,3))

    for i in range(len(x_i)-1):
        for j in range(len(y_i)-1):
            for k in range(len(z_i)-1):

                p_box = points[ (points[:,0]>=x_i[i]) & (points[:,0]<=x_i[i+1]) & (points[:,1]>=y_i[j]) & (points[:,1]<=y_i[j+1]) & (points[:,2]>=z_i[k]) & (points[:,2]<=z_i[k+1]) ]
                c_box = colors[ (points[:,0]>=x_i[i]) & (points[:,0]<=x_i[i+1]) & (points[:,1]>=y_i[j]) & (points[:,1]<=y_i[j+1]) & (points[:,2]>=z_i[k]) & (points[:,2]<=z_i[k+1]) ]

                

                pcd_box = create_pcd_from_points(p_box,c_box)
                bb = o3d.geometry.OrientedBoundingBox.get_axis_aligned_bounding_box(pcd_box)

                pcd_boxes.append(pcd_box) #stores open3d pcd geometry for points within the bounding box
                voxel_grids.append(bb) #stores bounding box open3d geometry



    #print("bounds x y z ",x_bounds, y_bounds, z_bounds)
    #print("number of voxel grids formed ",len(voxel_grids))

    return pcd_boxes, voxel_grids


def get_overdense_voxels(pcd_boxes):
    overdense = []
    for pcd_idx in range(len(pcd_boxes)):
        n = np.asarray(pcd_boxes[pcd_idx].points).shape[0]
        if n>2000:
            #print("overdense number ",n)
            overdense.append(pcd_idx)
    return overdense


def partition_voxelize(points, colors, point_density = 1000, pcd_boxes = [], voxel_grids = []):
    #keep partitioning voxel boxes until none of them contains more than a specified number of points
    n_partitions = 0
    pcd_boxes, voxel_grids = [], []
    pcd_boxes, voxel_grids = voxelize(points, colors, point_density = 1000, pcd_boxes = pcd_boxes, voxel_grids = voxel_grids)
    
    

    while(True):
        
        if len(voxel_grids)>n_partitions:
            n_partitions = len(voxel_grids)
        
        overdense_voxel_idcs = get_overdense_voxels(pcd_boxes)
        
        if overdense_voxel_idcs==[]:
            return pcd_boxes, voxel_grids

        else:
            pcd_boxes_new, voxel_grids_new = [], []
            
            for i in range(len(pcd_boxes)):
                if i in overdense_voxel_idcs:
                    p, c = np.asarray(pcd_boxes[i].points), np.asarray(pcd_boxes[i].colors)
                    pcd_boxes_sub, voxel_grids_sub = voxelize(p, c, point_density = 1000, pcd_boxes = [], voxel_grids = [])
                    
                    pcd_boxes_new.extend(pcd_boxes_sub)
                    voxel_grids_new.extend(voxel_grids_sub)
                else:
                    pcd_boxes_new.append(pcd_boxes[i])
                    voxel_grids_new.append(voxel_grids[i])

            pcd_boxes = pcd_boxes_new
            voxel_grids = voxel_grids_new

        

        if len(voxel_grids)==n_partitions:
            #print("paritions are not increasing anymore ")
            return pcd_boxes, voxel_grids

        #print("number of partitions ",len(voxel_grids))

def merge_small_voxels(pcd_boxes, voxel_grids):
    #keep merging small voxels till the size remains under a number of points
    idx_groups = []
    idcs = []
    N = 0

    for i in range(len(pcd_boxes)):
        
        n = np.asarray(pcd_boxes[i].points).shape[0]
        N+=n
        if N<2000:
            idcs.append(i)
        else:
            idx_groups.append(idcs)
            idcs = [i]
            N = n

    #append the last remaining groups
    idx_groups.append(idcs)

    #print("index groups ",idx_groups)
    #print("number of pcd boxes initially",len(pcd_boxes))

    pcd_boxes_new, voxel_grids_new = [],[]

    for group in idx_groups:
        pt,ct = np.empty((0,3)), np.empty((0,3))

        for g in group:
            p,c = np.asarray(pcd_boxes[g].points), np.asarray(pcd_boxes[g].colors)
            pt = np.concatenate((pt,p), axis=0)
            ct = np.concatenate((ct,c), axis=0)
        
        merge_pcd = create_pcd_from_points(pt,ct)

        bb = o3d.geometry.OrientedBoundingBox.get_axis_aligned_bounding_box(merge_pcd)
        pcd_boxes_new.append(merge_pcd) #stores open3d pcd geometry for points within the bounding box
        voxel_grids_new.append(bb) #stores bounding box open3d geometry

    #print("number of pcd boxes after merge ",len(pcd_boxes_new))
    #print("printing the number of points in each box ")
    for pcd in pcd_boxes_new:
        #print("size ",np.asarray(pcd.points).shape[0])
        pass

    return pcd_boxes_new, voxel_grids_new

