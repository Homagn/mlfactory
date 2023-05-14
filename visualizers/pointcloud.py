import open3d as o3d
import numpy as np
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import os
import sys


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

import math
import copy
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


from PIL import Image
import cv2


'''
#from nyu toolbox download/ camera_params.m
fx = 2*582.62 #multiply the value found in fx_d and fy_d by 2
fy = 2*582.69
centerX = 313.04
centerY = 238.44
scalingFactor = 5000 #10


#diode dataset
#from - https://github.com/diode-dataset/diode-devkit/blob/master/intrinsics.txt
#[fx, fy, cx, cy] = [886.81, 927.06, 512, 384]
fx = 2*886.81 #multiply the value found in fx_d and fy_d by 2
fy = 2*927.06
centerX = 512.0
centerY = 384.0
scalingFactor = 20000 #10
'''

def scale_camera_params(downsize_x, downsize_y, params):
    params["centerX"] = params["centerX"]/(downsize_x)
    params["centerY"] = params["centerY"]/(downsize_y)

    params["fx"] = params["fx"]/(downsize_x*2)
    params["fy"] = params["fy"]/(downsize_y*2)
    return params


# This is special function used for reading NYU pgm format
# as it is written in big endian byte order.
def read_nyu_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out

    

def show_pcd(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    #vis.add_geometry(cloud)
    vis.add_geometry(pcd)


    # run visualizer main loop
    print("Press Q or Excape to exit")
    vis.run()
    vis.destroy_window()

def get_rotation_matrix(transformations):
    rotx = transformations["rx"]
    roty = transformations["ry"]
    rotz = transformations["rz"]

    rotx,roty,rotz = math.radians(rotx),math.radians(roty),math.radians(rotz)

    Rx = np.array([ [1,0,0,0],
                    [0, math.cos(rotx), -math.sin(rotx), 0],
                    [0, math.sin(rotx), math.cos(rotx), 0],
                    [0,0,0,1]   ])

    Ry = np.array([ [math.cos(roty), 0, math.sin(roty), 0],
                    [0, 1, 0, 0],
                    [-math.sin(roty), 0, math.cos(roty), 0],
                    [0, 0, 0, 1]   ])

    Rz = np.array([ [math.cos(rotz), -math.sin(rotz), 0, 0],
                    [math.sin(rotz), math.cos(rotz), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]   ])
    R = np.matmul(Rz,Ry)
    R = np.matmul(R,Rx)

    return R

def point_transform(x,y,z, T):
    #homogenous transformation T
    points = np.array([[x,y,z,1]]).T

    T = np.array(T) #4x4 translation, rotation and scaling matrix

    points = T.dot(points)
    # Return all but last row
    return points[0],points[1],points[2]


def project_rgbd_to_pointcloud(rgb_file,depth_file, camera_params, transformations = {"rx":0,"ry":0,"rz":0} ):
    #ref - https://stackoverflow.com/questions/49598937/how-to-convert-the-depth-map-to-3d-point-clouds
    XS = []
    YS = []
    ZS = []
    C = []

    Transform_matrix = get_rotation_matrix(transformations)

    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file).convert('I')

    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")


    points = []    
    count = 0
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            print(count,end='\r')
            

            color = rgb.getpixel((u,v))
            Z = depth.getpixel((u,v)) / camera_params["scalingFactor"]
            #print(Z)
            if Z==0: 
                continue
            X = (u - camera_params["centerX"]) * Z / camera_params["fx"]
            Y = (v - camera_params["centerY"]) * Z / camera_params["fy"]
            #print("point details ",color,X,Y,Z)
            #example - point details  (150, 135, 133) -1.4616828358208955 0.6559148187633262 4.6086
            X,Y,Z = point_transform(X,Y,Z,Transform_matrix)

            XS.append(X)
            YS.append(Y)
            ZS.append(Z)
            clist = [color[0], color[1], color[2]]
            C.append(clist)
            count+=1
    print("num pcd points ",count)
    return XS,YS,ZS,C

def convert_to_open3dpointcloud(XS,YS,ZS,C):
    numpts = len(XS)
    xyz = np.zeros((len(XS), 3))
    colors = np.zeros((len(XS), 3))
    max_depth = np.max(ZS)

    for i in range (numpts):
        print(i,end='\r')
        #x, y, z = XS[i], YS[i], (255.0-ZS[i])*255

        #x, y= XS[i]/1024.0, YS[i]/728.0
        x, y= XS[i],  YS[i]
        z = ZS[i]
        #if x>-100 and x<=100 and y>-100 and y<=100 and z>-100 and z<=100:
        xyz[i,0] = x
        xyz[i,1] = y
        xyz[i,2] = z


        colors[i,:] = np.array([C[i][2], C[i][1], C[i][0]])/255.0
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd





def homogenous_transforms(pcd):
    #using native open3d functions for transforms of point clouds

    R = np.array([[ 0.707, -0.5,    0.5  ],
                 [ 0.0,     0.707,  0.707  ],
                 [ 0.707,  0.5,   -0.5,  ]])
    pcd = pcd.rotate(R, pcd.get_center())

    pcd = pcd.translate((2, 2, 2), relative=False)
    pcd = pcd.scale(0.5, center=pcd.get_center())

    return pcd


            
def show_pcd_from_rgbd(rgb, d, camera_params, save_loc = "", transformations = {"rx":0,"ry":0,"rz":0}):

    #uses brute force point by point construction of pointcloud 
    #useful when visualizing in other environments when raw pointcloud is needed as a numpy array
    #for example in jupyter notebooks

    cv2.imshow("rgb", rgb)
    cv2.waitKey(0)

    cv2.imshow("depth", d)
    cv2.waitKey(0)

    cv2.imwrite("rgb.png",rgb)
    cv2.imwrite("depth.png",d)

    print("Max of depth raw ",np.max(d))
    print("mean of depth raw ",np.mean(d))
    print("shape of depth raw ",d.shape)

    X,Y,Z,C = project_rgbd_to_pointcloud("rgb.png","depth.png", camera_params, transformations)
    pcd = convert_to_open3dpointcloud(X,Y,Z,C)

    #pcd = homogenous_transforms(pcd)

    show_pcd(pcd)
    if save_loc!="":
        print("writing pointcloud ")
        o3d.io.write_point_cloud(save_loc, pcd)



def show_pcd_from_rgbd_native(rgb, dep, camera_params, transformations = {"rx":0,"ry":0,"rz":0}, save_loc = ""):
    #uses open3d native functions to speedily construct pointcloud from rgb depth tuple

    fx, fy, cx, cy = camera_params["fx"], camera_params["fy"], camera_params["centerX"], camera_params["centerY"]
    width, height = camera_params["scalingFactor"], camera_params["scalingFactor"]

    # Obtain point cloud
    color = o3d.geometry.Image(rgb.astype(np.uint8))
    d = copy.copy(dep)

    md = np.max(d)
    print("max d",np.max(d))
    print("center point distance ",d[d.shape[0]//2, d.shape[1]//2])


    d /= np.max(d)

    

    depth = o3d.geometry.Image(d)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                                                              depth_scale=1.0,
                                                              depth_trunc=0.7,
                                                              convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    
    print("scaling pcd ")

    pcd = pcd.scale(md, center=(0,0,0)) 

    show_pcd(pcd)

    if save_loc!="":
        print("writing pointcloud ")
        o3d.io.write_point_cloud(save_loc, pcd)


if __name__ == "__main__":
    print("Read NYU dataset")
    # Open3D does not support ppm/pgm file yet. Not using o3d.io.read_image here.
    # MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.

    #color_raw = mpimg.imread("../../TestData/RGBD/other_formats/NYU_color.ppm")
    #depth_raw = read_nyu_pgm("../../TestData/RGBD/other_formats/NYU_depth.pgm")

    color_raw = mpimg.imread("/datasets/nyuv2/basements/basement_0001a/r-1316653580.484909-1316500621.ppm")
    depth_raw = read_nyu_pgm("/datasets/nyuv2/basements/basement_0001a/d-1316653580.471513-1316138413.pgm")

    nyu_camera_params = {
                        "fx": 2*582.62,
                        "fy": 2*582.69,
                        "centerX": 313.04,
                        "centerY": 238.44,
                        "scalingFactor": 5000 
                        }

    show_pcd_from_rgbd(color_raw, depth_raw, nyu_camera_params)
    



