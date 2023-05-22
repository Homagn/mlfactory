import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import numpy as np
from scipy.spatial.transform import Rotation as R

import time
import copy
import math

import os
import sys
from datetime import datetime as dt



# An installation agnostic method to find and link to root of the package which is mlfactory
#==========================================================
import re
try: #testing the functions locally without pip install
  import __init__
  cimportpath = os.path.abspath(__init__.__file__)
except: #testing while mlfactory is installed using pip
  import mlfactory
  cimportpath = os.path.abspath(mlfactory.__file__)+'/applications/deep_modular_scene_mapper/__init__.py'

idxlist = [m.start() for m in re.finditer(r"/", cimportpath)]
invoking_submodule = cimportpath[idxlist[-2]+1:idxlist[-1]]
print("In deep_modular_scene_mapper/main.py got invoking submodule using re",invoking_submodule)
main_package_loc = cimportpath[:cimportpath.rfind('mlfactory')+len('mlfactory')]
print("In deep_modular_scene_mapper/main.py got main package location ",main_package_loc)


os.environ['applications'] = main_package_loc+'/applications'
os.environ['top'] = main_package_loc
sys.path.append(os.path.join(os.environ['applications']))
sys.path.append(os.path.join(os.environ['top']))
#==========================================================



#from superglue_inference.match_pair import matcher

from applications.superglue_inference import match_pair
from applications.deep_modular_scene_mapper.tools.rigid_transform_3D import rigid_transform_3D
from applications.deep_modular_scene_mapper.tools import point_plane_icp
from applications.deep_modular_scene_mapper.tools.depth_estimator import monodepth

from applications.deep_modular_scene_mapper.tools import project_rgbd
from applications.deep_modular_scene_mapper.tools import sift_matching

#from tools import project_rgbd
#from tools import sift_matching





data_location = '/datasets/sample_videos/extracted/'
feature_matcher = match_pair.matcher()
md = monodepth()

#camera cannot move extremely fast there needs to be continuity so maximum change in angles 
#and translations must be bounded according to some assumptions on how the robot moves
geo_constraints = [500.0,500.0] # sum of squares of roll pitch yaw, sum of squares of translation x y and z

#try to remove overlapping pcd points while registering
combine_depth_feature_matching = False #uses sift like feature matching across depth images as well
viz_load_pairs = False

#gap between indices should be smaller when motion is faster

#for inddor movie
#first make sure to run read_video.py
#indices = [0, 30, 60, 80, 100, 120, 140, 160, 170, 180, 190, 200, 210, 220, 230, 235, 240, 250]

#indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140 ]
#indices = [140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270 ]
indices = [340, 350, 360, 370, 380, 390, 400, 410, 420, 430 ]


iphone_back_camera_params = {
                            "fx": 520.3,
                            "fy": 520.3,
                            "centerX": 320.0,
                            "centerY": 240.0,
                            "scalingFactor": 1
                        }


transform_sequence = []
correction_transforms = []

pcds = []
mapped_pcds = []
rgbds = []
depth_images = []

poses = [ np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ) ]

def create_pcd_from_points(points,colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def join_pcds(pcdlist):
  all_points = []
  all_colors = []


  for sub_pcd in pcdlist:
      p1 = np.asarray(sub_pcd.points)
      c1 = np.asarray(sub_pcd.colors)

      if all_points==[]:
          all_points = p1
          all_colors = c1
      else:
          all_points = np.concatenate((all_points,p1), axis=0)
          all_colors = np.concatenate((all_colors,c1), axis=0)

  pcd = create_pcd_from_points(all_points,all_colors)
  return pcd




print("estimating first frame depth ... ")
rgb1, depth1, max_d_init = md.predict_depth(data_location+str(indices[0])+".jpg")

print("rgb and depth image shapes ",rgb1.shape, depth1.shape)

for idx in range(len(indices)-1):
  print("time ",dt.now())

  index_pair = [indices[idx], indices[idx+1]]
  #===================================================================
  #load source and target rgbd images
  print("loading rgb images, predicting depths and making initial pcds for the pair ",index_pair)
  #rgbs,depths = project_rgbd.load_verify_extracted(data_location, index_pair)

  #===================================================================
  #(depth predictor module returns original image, its estimated depth and the constructed pcd)
  if idx!=0:
    rgb1, depth1, max_d1 = copy.copy(rgb2), copy.copy(depth2), copy.copy(max_d2)
  else:
    max_d1 = copy.copy(max_d_init)


  #rgb1, depth1, max_d1 = md.predict_depth(data_location+str(index_pair[0])+".jpg")
  rgb2, depth2, max_d2 = md.predict_depth(data_location+str(index_pair[1])+".jpg")

  
  #rescale depths back to absolute depths 
  absd1 = (max_d1/max_d_init)*depth1 
  absd2 = (max_d2/max_d_init)*depth2 

  depth_images.append(o3d.geometry.Image(absd1))
  #depth_o3d = o3d.geometry.Image(absd1)
  #image_o3d = o3d.geometry.Image( (255.0*(rgb1/np.max(rgb1))).astype('uint8') )
  #rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, depth_trunc=4.0, convert_rgb_to_intensity=False)
  #rgbds.append(rgbd_image)
  rgbds.append([rgb1, absd1])

  rgbs = [rgb1,rgb2]
  depths = [depth1,depth2]

  #project_rgbd.show_pcd([pcd1,pcd2])
  #===================================================================



  print("converting rgbd to pointclouds")
  pcd1, pixel_points1 = project_rgbd.project_rgbd_to_pointcloud(rgbs[0][:,:,::-1],(max_d1/max_d_init)*depths[0], iphone_back_camera_params)
  pcd2, pixel_points2 = project_rgbd.project_rgbd_to_pointcloud(rgbs[1][:,:,::-1],(max_d2/max_d_init)*depths[1], iphone_back_camera_params)

  #pcd1 = scale_pcd(pcd1)
  #pcd2 = scale_pcd(pcd2)

  #o3d.io.write_point_cloud(str(indices[idx])+".pcd", pcd1) #store the untransformed first pointcloud

  #===================================================================
  #show the source and target pointclouds
  if viz_load_pairs:
    project_rgbd.show_pcd([pcd1,pcd2])

















  #===================================================================
  #find pairs of matching pixel points between consequtive images (for ex using sift matching), randomly chose half of those matches
  #find the xyz point of these pairs by cross checking with depth map
  #use P3P algorithm to find rigid transformation
  #if wrong points are chosen as matching pixel points, the transformation is going to be abrupt and will probably violate geo constraints
  #if geo constraints are violated try again back from step 1, keep doing loop until geo constraints match
  inflate = 0
  for _ in range(10): #try atmost 10 times 

    #===================================================================
    #extract corresponding points using sift based feature matching in rgb domain
    print("extracting image features using superglue ")
    #s,d = sift_matching.compare_extra(data_location+str(index_pair[0])+".png", data_location+str(index_pair[1])+".png")

    i1,i2 = feature_matcher.prepare_images(rgbs[0], rgbs[1])
    s,d = feature_matcher.match(i1,i2, viz = False, ransac_refine = True)
    print("got number of matches rgb ",s.shape[0], d.shape[0])

    

    if combine_depth_feature_matching:
      d1,d2 = feature_matcher.prepare_images(depths[0],depths[1])
      s_,d_ = feature_matcher.match(d1,d2, viz = False, ransac_refine = True)
      print("got number of matches depth ",s_.shape[0], d_.shape[0])

      #trying to merge matching keypoints calculated on depth frames as well, not much benefit
      s = np.concatenate((s,s_))
      d = np.concatenate((d,d_))

    #maybe use lowes ratio test also?
    #lowes ration test explanation- https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work





    #===================================================================
    #get exact 3d corresponding points by mapping rgbs back to their pcds
    print("Doing rough alignment using least squares matching based on superglue matches")
    source_cor = []
    dest_cor = []

    n_frac_matches = 2
    if s.shape[0]<6:
      n_frac_matches = 1 #alg cannot work with too few matches

    rand_chose = np.random.choice(s.shape[0], s.shape[0]//n_frac_matches)

    

    #for i in range(s.shape[0]):
    for i in rand_chose:
      u,v = s[i]
      p,q = d[i]
      
      if list(pixel_points1[int(u),int(v)])[2]!=0.0 and list(pixel_points2[int(p),int(q)])!=0.0:
        source_cor.append( list(pixel_points1[int(u),int(v)]) )
        dest_cor.append( list(pixel_points2[int(p),int(q)]) )

    #===================================================================

    

    #print("3d source-dest correspondences ",source_cor, dest_cor)
    #===================================================================
    #estimate the rigid transform using SVD
    ret_R, ret_t = rigid_transform_3D(np.array(dest_cor).T, np.array(source_cor).T)
    print("estimated rotation and translation ",ret_R, ret_t)

    
    r = R.from_matrix(ret_R)
    pyr = r.as_euler('zyx', degrees=True)
    print("estimated pitch, yaw, roll change ",pyr)


    magn_r = 0.0
    magn_t = 0.0
    for e in pyr:
        magn_r+=e**2
    for e in ret_t:
        magn_t+= e**2
    print("got magnitudes of rot and trans transformations ",magn_r, magn_t)

    if magn_r<geo_constraints[0]+inflate and magn_t<geo_constraints[1]+inflate:
      print("transformation matched geometric motion constraint assumption ! exiting attempts")
      break
    else:
      print("proposed transformation did not match geo constraints !! ")
      inflate +=1

    #===================================================================












  #===================================================================
  #register destination pointcloud to source by rgb matched festures transformation and show the result
  print("Rough transform last pointcloud of the most recent pair")
  #roll all the pointclouds so far through all the prior transformations
  if transform_sequence!=[]:
    count = 1
    prev_pose = poses[0]
    for t in range(len(transform_sequence)):
      pcd1 = pcd1.transform(transform_sequence[t])
      pcd2 = pcd2.transform(transform_sequence[t])
      if len(poses)>2 and t!=len(transform_sequence)-1:
        poses[-1] = np.dot(poses[-1], transform_sequence[t])
  

  #apply current transformation
  T = np.eye(4)
  T[:3, :3] = ret_R
  T[0, 3] = ret_t[0][0]
  T[1, 3] = ret_t[1][0]
  T[2, 3] = ret_t[2][0]

  pcd2 = pcd2.transform(T)
  print("Now showing registered pcds ")
  #project_rgbd.show_pcd([pcd1,pcd2])
  #poses.append(poses[-1])
  transform_sequence.append(T)
  poses.append(T)
  #poses[-1] = np.dot(poses[-1], T)




  #===================================================================
  #remove overlapping points between source and target (as much as possible)
  #lot of points should be removed from pcd1 based on overlap with target(pcd2)
  #===================================================================
  #paiwise registration refinement
  

  if pcds==[]:
    pcds.append(pcd1)
    pcds.append(pcd2)

  else:
    pcds[-1] = pcd1
    pcds.append(pcd2)

  #o3d.io.write_point_cloud(str(indices[idx])+"_T.pcd", pcds[idx])

  
  rpcd1, rpcd2, correction_transform = point_plane_icp.register_pair(pcds[-2],pcds[-1], viz = False)
  if mapped_pcds==[]:
    mapped_pcds.append(rpcd1)
    mapped_pcds.append(rpcd2)

  else:
    
    mapped_pcds[-1] = rpcd1
    mapped_pcds.append(rpcd2)
  correction_transforms.append(correction_transform)
  
  #poses[-1] = np.matmul(poses[-1], correction_transform)

  #backpropagate the last correction transform to all previous mapped pcds other than the most recent pair
  if correction_transforms!=[]:
      for i in range(len(mapped_pcds)-2):
        mapped_pcds[i].transform(correction_transforms[-1])
        poses[i] = np.dot(poses[i], correction_transforms[-1])

  #===================================================================




print("all the feature matching transform sequence ",len(transform_sequence), transform_sequence)
print("all the icp correction transforms ",len(correction_transforms), correction_transforms)



#===================================================================
#(save individual point clouds and join pcds into scenemap)
'''
print("saving pcd files ")
for idx in range(len(mapped_pcds[:-1])):
  o3d.io.write_point_cloud(str(idx)+".pcd", mapped_pcds[idx]) 
'''


scenemap = join_pcds(mapped_pcds[:-1])
print("showing all the mapped registered pointclouds except last one")
vis_cam_param = project_rgbd.show_pcd([scenemap], return_camera_params = True)
#===================================================================





#===============================================================
#(multiway registration over multiple roughly aligned point clouds)
#(again get the scenemap but for multiway registration using pose graph - slightly better aligned)
from tools import pose_graph_optimizer

refined_pcds, refined_poses = pose_graph_optimizer.run(mapped_pcds[:-1])

print("Refined pcd poses now showing them")
project_rgbd.show_pcd(refined_pcds)


scenemap = join_pcds(refined_pcds)
print("downsampling final map")
scenemap = scenemap.voxel_down_sample(voxel_size=0.015)
#===================================================================





#===================================================================
#(visualize change in poses as coord meshes)
coord_meshes = []
#poses = poses[1:]
for i in range(len(poses)-1):
  poses[i] = np.dot(poses[i], refined_poses[i])
  
  coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
  coord_mesh.scale(0.1, center=coord_mesh.get_center()) 
  coord_mesh.transform(poses[i])
  #coord_mesh.transform(refined_poses[i])
  coord_meshes.append(coord_mesh)
#===================================================================




#===================================================================
#(show the pose graph optimized scene along with pose changes as coord meshes)
print("got pose sequence ",poses)
scene_with_coords = [scenemap]
scene_with_coords.extend(coord_meshes)
project_rgbd.show_pcd(scene_with_coords)
print("Total number of points in the scene ",np.array(scenemap.points).shape[0])
#===============================================================






#===================================================================
#(check the accuracy of poses by reconstructing the scene from untransformed pointclouds stored earlier in rgbds list)
#(using only the pose list)
print("Now trying to get back scene from rgbd and poses ...")
opcds=[]
#poses.insert(0, np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ))
for i in range(len(poses)-1):
  pcd, _ = project_rgbd.project_rgbd_to_pointcloud(rgbds[i][0][:,:,::-1],rgbds[i][1], iphone_back_camera_params)
  pcd = pcd.transform(poses[i])
  opcds.append(pcd)

project_rgbd.show_pcd(opcds)
#===================================================================







'''
to do

[
application of opencv solvepnp - 
given now we have a full map of the room (stored in form of sequence of rgb,depth and pose/ and a compact simplified mesh for the entire scene M), 
we can get an rgb 
image at any location in the room
scan over all the rgbd, pose pairs in the map and find the rgb image which has highest match with the image at that location
say the matching rgb image is rgb_m and the current image is rgb_i
find all the 2d correspondences between rgb_m and rgb_i 
also we can retrieve all the 3d points of the 2d correspondence from the depth counterpart of rgb_m

using opencv solve pnp (2d points, 3d points) we can get the rotation and translation of rgb_i with respect to rgb_m

This technique can be used for real time localization as well over the simplified map M
]



Depth reprojection based corrections :
(see the reproject_depth_map function in depth_estimator.py )
(collection of depth maps can be corrected maybe using some sort of averaging and reprojection)
'''