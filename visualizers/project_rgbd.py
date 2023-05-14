import open3d as o3d
import numpy as np
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import os
import sys


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




def reproject_depth_map(pcd):
    #========================================================================
    #depth reprojection part
    print("Now saving depth image reprojection of the pcd as depth.png...")
    img_width, img_height = (2048, 128)
    #pcd = o3d.io.read_triangle_mesh('bunny.ply')
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)

    pcd.estimate_normals()
    pcd.normalize_normals()


    '''
    #apply perspective transformation of another camera pose, say the pose of that camera is trans, but you want 
    #to look at current view as if it was being viewed from that pose

    dtype = o3d.core.float32
    trans = np.array([[ 9.91051327e-01,  5.56196799e-02, -1.21341332e-01,
                        -1.64767893e-02],
                       [-5.60779272e-02,  9.98426329e-01, -3.62212664e-04,
                         1.46362140e-01],
                       [ 1.21130234e-01,  7.16354171e-03,  9.92610775e-01,
                        -7.51683117e-02],
                       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                         1.00000000e+00]])
    trans_inv = np.linalg.inv(trans)
    
    transformation = o3d.core.Tensor(trans_inv, dtype)
    pcd.transform(transformation)
    '''
    



    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'

    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
    renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
    #renderer_pc.scene.add_geometry("pcd", pcd, mat)
    renderer_pc.scene.add_geometry("__model__",pcd, mat)
    #renderer_pc.scene.add_geometry(pcd)

    print("current camera view matrix ",renderer_pc.scene.camera.get_view_matrix())




    depth_image = np.asarray(renderer_pc.render_to_depth_image())
    np.save('depth', depth_image)
    normalized_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())

    print("depth image ",depth_image.shape, np.mean(depth_image))
    cv2.imshow("depth image ",normalized_image)
    cv2.waitKey(0)

    #normalized_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    #normalized_image = depth_image
    plt.imshow(normalized_image, cmap='gray', vmin=0, vmax=1)
    plt.savefig('depth.png')




def show_pcd(pcds, return_camera_params = False):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #vis.create_window(width=832, height=448)
    

    #vis.add_geometry(cloud)
    for pcd in pcds:
        vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # run visualizer main loop
    print("Press Q or Excape to exit")
    vis.run()
    

    vis.destroy_window()
    if return_camera_params:
        return param


def correction(A):
  #far away things become more farther than close away things
  max_A = 255.0
  C = A/max_A

  B = (math.pi/3)*(A/max_A)  #bring in the range of 0 and pi/3 in the tan graph
  B[B==0]=0.1

  #print("correction mask ",np.tan(B)/C)
  #print("correction mask inverse ",C/np.tan(B))
  #print("correction mask inverse result ",(C/np.tan(B))*A)

  correction_increase = np.tan(B)*max_A
  correction_decrease = (C/np.tan(B))*A

  return correction_increase, correction_decrease


def project_rgbd_to_pointcloud(rgb_file,depth_file, camera_params, normalizer = 255.0 ): #we divide the pcd points in the end using normalizer
    #ref - https://stackoverflow.com/questions/49598937/how-to-convert-the-depth-map-to-3d-point-clouds

    if isinstance(rgb_file,str) and isinstance(depth_file,str):
        rgb = cv2.imread(rgb_file)
        depth = cv2.imread(depth_file,0)
    else:
        '''
        cv2.imwrite("rgb.png",rgb_file)
        cv2.imwrite("depth.png",depth_file)
        rgb = cv2.imread("rgb.png")

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread("depth.png",0)
        '''
        try:
            rgb = copy.copy(cv2.cvtColor(rgb_file, cv2.COLOR_BGR2RGB))
        except:
            rgb = copy.copy(rgb_file)
            print("WARN ! couldnt convert bgr to rgb")
        depth = copy.copy(depth_file)


    
    


    color = rgb
    #get the Z coordinates of the pointcloud ordered by the pixel locations in rgb image (depth image)
    Z = depth/camera_params["scalingFactor"]
    #Z = depth/np.max(depth)

    
    #not sure which one to use of zi or zd or dont even use correction
    #zi,zd = correction(Z)
    #Z= np.concatenate((zd[:128,:],zi[128:,:]))
    #Z = zi


    #print("max and min of Z ",np.max(Z), np.min(Z))

    imgrid = np.mgrid[0:rgb.shape[0],0:rgb.shape[1]] 
    #get the X and Y coordinates of the pointcloud ordered by the pixel locations in rgb image

    mask = np.where(Z!=0.0, Z, 0.0)
    #mask = np.where(Z==0.0, Z, 1.0)

    X =  (imgrid[1,:,:] - camera_params["centerX"]) * mask / camera_params["fx"] 
    Y =  (imgrid[0,:,:] - camera_params["centerY"]) * mask / camera_params["fy"] 

    
    zero_points = np.where(Z!=0)[0]
    print("where zero ",np.where(X==0),np.where(Y==0),np.where(Z==0))
    np.delete(X, zero_points, axis=0)
    np.delete(Y, zero_points, axis=0)
    np.delete(Z, zero_points, axis=0)
    
    

    #X =  (imgrid[1,:,:] - camera_params["centerX"])  / camera_params["fx"] 
    #Y =  (imgrid[0,:,:] - camera_params["centerY"])  / camera_params["fy"] 



    
    print("X Y Z shapes ",X.shape,Y.shape,Z.shape)

    #convert X,Y,Z points to a pcd
    xyz = np.dstack([X,Y,Z]).reshape((X.shape[0]*X.shape[1],3))/ normalizer
    print("XYZ shape ",xyz.shape)
    colors = color.reshape((X.shape[0]*X.shape[1],3))/255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)


    #rotate the pcd along x axis by 180 degrees to get normal orientation pcd
    R = pcd.get_rotation_matrix_from_xyz((np.pi , 0, 0))
    pcd = pcd.rotate(R, center=(0,0,0))


    #recover rotated X,Y,Z from the rotated pcd
    rpts = np.array(pcd.points).reshape((X.shape[0],X.shape[1],3))
    X,Y,Z = rpts[:,:,0], rpts[:,:,1], rpts[:,:,2]
    print("X Y Z shapes ",X.shape,Y.shape,Z.shape)
    colors = np.array(pcd.colors).reshape((X.shape[0],X.shape[1],3))*255.0
    

    #for each pixel location in the RGB image stores the corresponding 3d coordinate it is mapped to
    pixel_points = np.dstack([X.T,Y.T,Z.T])
    print("pixel points shape ",pixel_points.shape)
    return pcd, pixel_points


def pcd_from_rgbd_native(rgb, dep, camera_params):
    #uses open3d native functions to speedily construct pointcloud from rgb depth tuple

    fx, fy, cx, cy = camera_params["fx"], camera_params["fy"], camera_params["centerX"], camera_params["centerY"]
    width, height = camera_params["scalingFactor"], camera_params["scalingFactor"]

    # Obtain point cloud
    color = o3d.geometry.Image(rgb.astype(np.uint8))
    #d = copy.copy(dep)
    d = dep.astype(np.uint8)

    md = np.max(d)
    #print("max d",np.max(d))
    #print("center point distance ",d[d.shape[0]//2, d.shape[1]//2])


    #d /= np.max(d)

    

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
    
    #print("scaling pcd ")

    pcd = pcd.scale(md, center=(0,0,0)) 

    return pcd

            
def show_pcd_from_rgbd(rgb, d, camera_params, save_loc = ""):

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

    #X,Y,Z,C, pixel_points = project_rgbd_to_pointcloud("rgb.png","depth.png", camera_params, transformations)
    pcd, pixel_points = project_rgbd_to_pointcloud("rgb.png","depth.png", camera_params)

    #pcd = convert_to_open3dpointcloud(X,Y,Z,C)

    #pcd = homogenous_transforms(pcd)

    show_pcd([pcd])
    if save_loc!="":
        print("writing pointcloud ")
        o3d.io.write_point_cloud(save_loc, pcd)


def load_verify_extracted(images_folder, indices, im_resize = (256,256)):
  rgbs = []
  depths = []
  video_extract_loc = images_folder

  def process_imsave_name(count):
    image_index = 1000+count #trying to keep constant number of characters
    image_index = str(image_index)
    
    image_index = '0'+image_index[1:]
    image_index = image_index
    return image_index


  for c in indices:
    img_rgb = cv2.imread(video_extract_loc+"rgb/"+process_imsave_name(c)+".png")
    img_dep = cv2.imread(video_extract_loc+"depth/"+process_imsave_name(c)+".png")


    desired_size = im_resize#was originally 256, trim borders made it 216

    img_rgb = cv2.resize(img_rgb, desired_size, interpolation = cv2.INTER_AREA)
    img_dep = cv2.resize(img_dep, desired_size, interpolation = cv2.INTER_AREA)

    print("project rgbd image shape ",img_rgb.shape) #(1504, 1128, 3)

    img_dep = cv2.cvtColor(img_dep, cv2.COLOR_BGR2GRAY) #make sure depth is read grayscale
    depths.append(img_dep)

    rgbs.append(img_rgb)

  return rgbs,depths




if __name__ == "__main__":
    indices = [200]

    rgbs,depths = load_verify_extracted('data/table/', indices)

    iphone_front_camera_params = {
                            "fx": 2*118.62,
                            "fy": 2*99.79,
                            "centerX": 165.91,
                            "centerY": 175.91,
                            "scalingFactor": 1
                        }

    show_pcd_from_rgbd(rgbs[0], depths[0], iphone_front_camera_params, transformations = {"rx":180,"ry":0,"rz":0}, save_loc = "1.pcd")








'''

sample X,Y,Z  -115.39011127971673 -145.4311554263954 165.0
sample X,Y,Z  -113.9994941831057 -144.54975448441726 164.0
sample X,Y,Z  -113.30821109425054 -144.54975448441726 164.0
sample X,Y,Z  -113.30361659079412 -145.4311554263954 165.0
sample X,Y,Z  -113.97306525037936 -147.19395731035175 167.0
sample X,Y,Z  -114.62565334682178 -148.95675919430803 169.0
sample X,Y,Z  -113.9132945540381 -148.95675919430803 169.0
sample X,Y,Z  -114.54059180576631 -150.71956107826435 171.0
sample X,Y,Z  -114.48541561288147 -151.6009620202425 172.0
sample X,Y,Z  -115.08320687910975 -153.3637639041988 174.0
sample X,Y,Z  -116.32132018209407 -156.00796673013326 177.0
sample X,Y,Z  -116.22820772213791 -156.88936767211143 178.0
sample X,Y,Z  -116.77541729893778 -158.65216955606772 180.0
sample X,Y,Z  -117.30576631259483 -160.41497144002403 182.0
sample X,Y,Z  -117.81925476310907 -162.17777332398035 184.0
sample X,Y,Z  -117.04366885853986 -162.17777332398035 184.0
sample X,Y,Z  -117.5318664643399 -163.94057520793666 186.0
sample X,Y,Z  -117.3755268925982 -164.8219761499148 187.0
sample X,Y,Z  -117.83421851289832 -166.58477803387112 189.0
sample X,Y,Z  -117.03755690440059 -166.58477803387112 189.0






 [[ 0.9809229  -0.08412022  0.17525423]
 [ 0.08599556  0.99629063 -0.00312021]
 [-0.17434168  0.01813177  0.98451827]] [[-3.32887237]
 [ 0.33068204]
 [-1.95098305]]

'''