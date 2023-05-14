import cv2
import numpy as np
import open3d as o3d
import math

import sys
import os
import copy

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)


    vis = o3d.visualization.Visualizer()
    vis.create_window()

    #vis.add_geometry(cloud)
    vis.add_geometry(source_temp)
    vis.add_geometry(target)

    #opt = vis.get_render_option()
    #opt.show_coordinate_frame = True

    
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.03, origin = (0,0,0))
    vis.add_geometry(mesh)

    mesh2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.03, origin = source.get_center()-target.get_center())
    vis.add_geometry(mesh2)
    


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


def homogenous_transforms(R, T, pcd, pitch = 0.0, yaw = 0.0):
    #using native open3d functions for transforms of point clouds

    #R = R[0:3,0:3] #convert to 3x3
    #pcd = pcd.rotate(R, center=(0,0,0))

    print("In homogeneous transforms ->rotating along axes ..")

    #R = pcd.get_rotation_matrix_from_yzx((math.radians(pitch), math.radians(yaw), math.radians(0)))
    #R = pcd.get_rotation_matrix_from_axis_angle((math.radians(pitch), math.radians(yaw), math.radians(0)))
    R = pcd.get_rotation_matrix_from_xyz(np.array([math.radians(pitch), math.radians(yaw), math.radians(0)]))
    pcd = pcd.rotate(R, center=(0, 0, 0))

    print("In homogeneous transforms ->translating ..")
    pcd = pcd.translate(T, relative=True)

    #pcd = pcd.transform(R)


    return pcd

def compensate_pcd_creation_scaling(pcd, ratio = 1.0):
    pcd = pcd.scale(ratio, center=(0,0,0)) #1.055
    return pcd


def known_transform_align(pcd_source, pitch = 0, yaw = 0, roll = 0, sideways = 0,upwards = 0, forward = 0):
    transformations = {"rx":pitch, "ry":yaw, "rz":roll}

    R = get_rotation_matrix(transformations)
    print("got R ",R)

    
    T = (sideways,upwards,forward)
    print("sending T ",T)

    pcd_transformed = homogenous_transforms(R,T,pcd_source, pitch=pitch, yaw=yaw)
    return pcd_transformed



def register(source_name,target_name, pitch_change = 0.0, yaw_change = 0.0, sideways_change = 0.0, upwards_change = 0.0, forward_change = 0.0):

    print("1. Load two point clouds and show initial pose")
    source = o3d.io.read_point_cloud(source_name)
    target = o3d.io.read_point_cloud(target_name)

    # draw initial alignment
    current_transformation = np.identity(4)
    print("original unaligned...")
    draw_registration_result_original_color(source, target, current_transformation)

    xd,yd,zd = source.get_center()-target.get_center()
    print("source target center difference ",source.get_center()-target.get_center())

    
    '''
    #probably need this additional transform because of the relative location of the camera to the drone frame
    transformations = {"rx":-pitch_change, "ry":-yaw_change, "rz":0}
    R = get_rotation_matrix(transformations)
    dx,dy,dz = point_transform(0.015,0.015,0.015,R)
    target = known_transform_align(target, pitch = 0, yaw= 0, sideways = -dx,upwards = dy,forward = -dz)
    '''
    
    


    #finally apply the known angle transitions from aithor sim
    target_s = known_transform_align(target, pitch = -pitch_change, yaw= -yaw_change, sideways = 0,upwards = 0,forward = 0)
    #upwards change is not '-' sign because open3d aligns the upwards axis in a reverse direction
    target_ss = known_transform_align(target_s, pitch = 0.0, yaw = 0.0, sideways = -sideways_change,upwards = upwards_change,forward = -forward_change)
    
    #target_ss = known_transform_align(target_ss, pitch = 0, yaw= 10, sideways = 0,upwards = 0,forward = 0)

    print("check if final pcd nan ",np.array(target_ss.points))
    


    print("final aligned ... ")
    draw_registration_result_original_color(source, target_ss, current_transformation)

    print("source target center difference ",source.get_center()-target_ss.get_center())


if __name__ == '__main__':

    #pitch_change = 2.pcd pitch - 1.pcd pitch/ same with translations and other rotations

    register("1.pcd","2.pcd", 
            pitch_change = 10.0,
            yaw_change = -10.0,
            sideways_change = -0.1,
            upwards_change = -0.1,
            forward_change = 0.1)


    '''
    register("1.pcd","2.pcd", 
            pitch_change = 10.0,
            yaw_change = 10.0,
            sideways_change = -0.1,
            upwards_change = -0.1,
            forward_change = 0.1)
    '''

    '''
    register("1.pcd","2.pcd", 
            pitch_change = 0.0,
            yaw_change = 0.0,
            sideways_change = 0.0,
            upwards_change = 0.0,
            forward_change = -0.3)
    '''