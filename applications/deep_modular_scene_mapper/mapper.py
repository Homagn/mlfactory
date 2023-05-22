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
  if 'extensions' in cimportpath:
    print("Non local testing ")
    import mlfactory
    cimportpath = os.path.abspath(mlfactory.__file__)+'/applications/deep_modular_scene_mapper/__init__.py'

except: #testing while mlfactory is installed using pip
  print("Non local testing")
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
from applications.deep_modular_scene_mapper.tools import pose_graph_optimizer


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


class mapper(object):
  def __init__(self, data_location = '/datasets/sample_videos/extracted/', frame_numbers = [0,10,20,30], init_pose = np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ) ):
    self.data_location = data_location
    self.feature_matcher = match_pair.matcher()
    self.md = monodepth() #the network that estimates mono depth
    self.voxel_size = 0.01
    self.init_pose = init_pose

    #camera cannot move extremely fast there needs to be continuity so maximum change in angles 
    #and translations must be bounded according to some assumptions on how the robot moves
    self.geo_constraints = [500.0,500.0] # sum of squares of roll pitch yaw, sum of squares of translation x y and z

    #try to remove overlapping pcd points while registering
    self.combine_depth_feature_matching = False #uses sift like feature matching across depth images as well
    self.viz_load_pairs = False

    #gap between indices should be smaller when motion is faster
    #self.indices = [340, 350, 360, 370, 380, 390, 400, 410, 420, 430 ]
    self.indices = frame_numbers


    self.iphone_back_camera_params = {
                                "fx": 520.3,
                                "fy": 520.3,
                                "centerX": 320.0,
                                "centerY": 240.0,
                                "scalingFactor": 1
                            }


    self.transform_sequence = []
    self.correction_transforms = []

    self.pcds = []
    self.mapped_pcds = [] #stores the final perfectly aligned map
    self.pose_sequence = [] #stores the final perfectly aligned poses
    self.rgbds = []
    self.depth_images = []

    #initialize poses
    self.poses = [ np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ) ]
    


    print("estimating first frame depth ... ")
    self.rgb1, self.depth1, self.max_d_init = self.md.predict_depth(self.data_location+str(self.indices[0])+".jpg")

    print("rgb and depth image shapes ",self.rgb1.shape, self.depth1.shape)

  def estimate_pointcloud_t1t2(self, idx):
    print("time ",dt.now())

    self.index_pair = [self.indices[idx], self.indices[idx+1]]
    #===================================================================
    #load source and target rgbd images
    print("loading rgb images, predicting depths and making initial pcds for the pair ",self.index_pair)
    #rgbs,depths = project_rgbd.load_verify_extracted(data_location, index_pair)

    #===================================================================
    #actual resized rgb image, estimated depth , and also the maximum value of depth according to a global scale
    if idx!=0:
      self.rgb1, self.depth1, self.max_d1 = copy.copy(self.rgb2), copy.copy(self.depth2), copy.copy(self.max_d2)
    else:
      self.max_d1 = copy.copy(self.max_d_init)

    #estimate for the adjacent time step
    self.rgb2, self.depth2, self.max_d2 = self.md.predict_depth(self.data_location+str(self.index_pair[1])+".jpg")

    
    #rescale depths back to absolute depths 
    self.absd1 = (self.max_d1/self.max_d_init)*self.depth1 
    self.absd2 = (self.max_d2/self.max_d_init)*self.depth2 

    self.depth_images.append(o3d.geometry.Image(self.absd1))
    self.rgbds.append([self.rgb1, self.absd1])

    self.rgbs = [self.rgb1,self.rgb2]
    self.depths = [self.depth1,self.depth2]

    #project_rgbd.show_pcd([pcd1,pcd2])
    #===================================================================


    #end result of this step- get the pointclouds that now need to be aligned
    print("converting rgbd to pointclouds")
    self.pcd1, self.pixel_points1 = project_rgbd.project_rgbd_to_pointcloud(self.rgbs[0][:,:,::-1],(self.max_d1/self.max_d_init)*self.depths[0], self.iphone_back_camera_params)
    self.pcd2, self.pixel_points2 = project_rgbd.project_rgbd_to_pointcloud(self.rgbs[1][:,:,::-1],(self.max_d2/self.max_d_init)*self.depths[1], self.iphone_back_camera_params)
    
    #initialize poses of the pointclouds
    #self.pose1 = np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ) 
    #self.pose2 = np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ) 
    
    self.pose1 = self.init_pose
    self.pose2 = self.init_pose
    #diagnostic
    if self.viz_load_pairs:
      project_rgbd.show_pcd([self.pcd1,self.pcd2])
  
  def find_rgb_feature_matchest1t2(self):
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

      i1,i2 = self.feature_matcher.prepare_images(self.rgbs[0], self.rgbs[1])
      s,d = self.feature_matcher.match(i1,i2, viz = False, ransac_refine = True)
      print("got number of matches rgb ",s.shape[0], d.shape[0])

      

      if self.combine_depth_feature_matching: #also add sift like feature match between mono depth estimated images (does not generally work well)
        d1,d2 = self.feature_matcher.prepare_images(self.depths[0],self.depths[1])
        s_,d_ = self.feature_matcher.match(d1,d2, viz = False, ransac_refine = True)
        print("got number of matches depth ",s_.shape[0], d_.shape[0])

        #trying to merge matching keypoints calculated on depth frames as well, not much benefit
        s = np.concatenate((s,s_))
        d = np.concatenate((d,d_))


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
        
        if list(self.pixel_points1[int(u),int(v)])[2]!=0.0 and list(self.pixel_points2[int(p),int(q)])!=0.0:
          source_cor.append( list(self.pixel_points1[int(u),int(v)]) )
          dest_cor.append( list(self.pixel_points2[int(p),int(q)]) )

      #===================================================================



      #This part uses the feature matches from rgb and finds the xyz in the corresponding pointclouds
      #using pairs of xyz matches now we can use P3P algorithm to find rough rotation and translation
      #===================================================================
      #estimate the rigid transform using SVD
      ret_R, ret_t = rigid_transform_3D(np.array(dest_cor).T, np.array(source_cor).T)
      print("estimated rotation and translation ",ret_R, ret_t)
      #===================================================================
      #store the final best rot and trans as the end result of this function
      self.ret_R, self.ret_t = ret_R, ret_t

      
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

      if magn_r<self.geo_constraints[0]+inflate and magn_t<self.geo_constraints[1]+inflate:
        print("transformation matched geometric motion constraint assumption ! exiting attempts")
        break
      else:
        print("proposed transformation did not match geo constraints !! ")
        inflate +=1

      





  def align_pointcloudst1t2(self):
    #===================================================================
    #register destination pointcloud to source by rgb matched festures transformation and show the result
    print("Rough transform last pointcloud of the most recent pair")
    #roll all the pointclouds so far through all the prior transformations
    if self.transform_sequence!=[]:
      count = 1
      prev_pose = self.poses[0]
      for t in range(len(self.transform_sequence)):
        self.pcd1 = self.pcd1.transform(self.transform_sequence[t])
        self.pcd2 = self.pcd2.transform(self.transform_sequence[t])

        self.pose1 = np.dot(self.pose1, self.transform_sequence[t])
        self.pose2 = np.dot(self.pose2, self.transform_sequence[t])

        if len(self.poses)>2 and t!=len(self.transform_sequence)-1:
          self.poses[-1] = np.dot(self.poses[-1], self.transform_sequence[t])
    

    #apply current transformation
    T = np.eye(4)
    T[:3, :3] = self.ret_R
    T[0, 3] = self.ret_t[0][0]
    T[1, 3] = self.ret_t[1][0]
    T[2, 3] = self.ret_t[2][0]

    self.pcd2 = self.pcd2.transform(T)
    self.pose2 = np.dot(self.pose2, T)

    print("Now showing registered pcds ")
    #project_rgbd.show_pcd([pcd1,pcd2])
    #poses.append(poses[-1])
    self.transform_sequence.append(T)
    self.poses.append(T)
    #poses[-1] = np.dot(poses[-1], T)




    if self.pcds==[]:
      self.pcds.append(self.pcd1)
      self.pcds.append(self.pcd2)

      

    else:
      self.pcds[-1] = self.pcd1
      self.pcds.append(self.pcd2)

      

    #o3d.io.write_point_cloud(str(indices[idx])+"_T.pcd", pcds[idx])

    
    #further refine the alignment by doing additional ICP step for the two pointclouds
    rpcd1, rpcd2, correction_transform = point_plane_icp.register_pair(self.pcds[-2],self.pcds[-1], viz = False)
    
    self.pose1 = np.dot(self.pose1, correction_transform)
    if self.mapped_pcds==[]:
      self.mapped_pcds.append(rpcd1)
      self.mapped_pcds.append(rpcd2)

      self.pose_sequence.append(self.pose1)
      self.pose_sequence.append(self.pose2)

    else:
      
      self.mapped_pcds[-1] = rpcd1
      self.mapped_pcds.append(rpcd2)

      self.pose_sequence[-1] = self.pose1
      self.pose_sequence.append(self.pose2)


    self.correction_transforms.append(correction_transform)
    
    #poses[-1] = np.matmul(poses[-1], correction_transform)

    #backpropagate the last correction transform to all previous mapped pcds other than the most recent pair
    if self.correction_transforms!=[]:
        for i in range(len(self.mapped_pcds)-2):
          self.mapped_pcds[i].transform(self.correction_transforms[-1])
          self.poses[i] = np.dot(self.poses[i], self.correction_transforms[-1])
          self.pose_sequence[i] = np.dot(self.pose_sequence[i],self.correction_transforms[-1])

  def run_sequence(self):
    for idx in range(len(self.indices)-1):
      self.estimate_pointcloud_t1t2(idx)
      self.find_rgb_feature_matchest1t2()
      self.align_pointcloudst1t2()

    print("all the feature matching transform sequence ",len(self.transform_sequence), self.transform_sequence)
    print("all the icp correction transforms ",len(self.correction_transforms), self.correction_transforms)

  def global_refinement(self):
    self.refined_pcds, self.refined_poses = pose_graph_optimizer.run(self.mapped_pcds[:-1])
    for i in range(len(self.poses)-1):
      self.pose_sequence[i] = np.dot(self.pose_sequence[i], self.refined_poses[i])
    

  

  def merge_map_poses(self, display_result = 'colab'):
    #===================================================================
    #(check the accuracy of poses by reconstructing the scene from untransformed pointclouds stored earlier in rgbds list)
    #(using only the pose list)
    print("Now trying to get back scene from rgbd and poses ...")
    opcds=[]
    #poses.insert(0, np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ))
    for i in range(len(self.poses)-1):
      pcd, _ = project_rgbd.project_rgbd_to_pointcloud(self.rgbds[i][0][:,:,::-1],self.rgbds[i][1], self.iphone_back_camera_params)
      #pcd = pcd.transform(self.poses[i])
      pcd = pcd.transform(self.pose_sequence[i])
      #pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
      opcds.append(pcd)
    #===================================================================
    self.scene_map = join_pcds(opcds)
    self.scene_map = self.scene_map.voxel_down_sample(voxel_size=self.voxel_size)

    #===================================================================
    #(visualize change in poses as coord meshes)
    self.coord_meshes = []
    #poses = poses[1:]
    for i in range(len(self.poses)-1):
      #self.pose_sequence[i] = np.dot(self.pose_sequence[i], self.refined_poses[i])
      
      coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
      coord_mesh.scale(0.1, center=coord_mesh.get_center()) 
      coord_mesh.transform(self.pose_sequence[i])
      #coord_mesh.transform(refined_poses[i])
      self.coord_meshes.append(coord_mesh)
    
    #self.scene_map.extend(self.coord_meshes)
    self.scene_with_coords = [self.scene_map]
    self.scene_with_coords.extend(self.coord_meshes)
    if display_result=='colab':
      project_rgbd.show_pcd_colab(self.scene_with_coords)
    elif display_result=='dont_display':
      print("finished mapping, check results by inspecting mapper.scene_with_coords")
    else:
      project_rgbd.show_pcd(self.scene_with_coords)



    #project_rgbd.show_pcd([self.scene_map, self.coord_meshes])
    print("Total number of points in the scene ",np.array(self.scene_map.points).shape[0])
    #===================================================================


if __name__ == '__main__':
  '''
  f = [240, 250, 260, 270, 280, 290, 300, 310, 320, 330 ,
       340, 350, 360, 370, 380, 390, 400, 410, 420, 430 , 
       440, 450, 460, 470, 480, 490, 500, 510, 520, 530 ]
  '''
  f = [340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360]

  #f = [340, 350, 360, 370, 380, 390, 400, 410, 420, 430] #map starts to lose details and gets highly overlapped on increasing number of frames
  #initial unit pose
  i_pose = np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ) 

  m = mapper(frame_numbers = f, init_pose = i_pose)
  m.voxel_size = 0.01
  m.run_sequence()
  m.global_refinement()
  m.merge_map_poses()


  ### to do
  #==============
  '''
  0. decide a global maximum depth value to normalize the depth map and not the max of the first depth map
  1. instead of final voxel downsample maintain an iterative map by downsampling iteratively and merging new points iteratively
  2. superglue matching, replace P3P alignment with fundamental matrix based alignment (dont use depth estimation points) - additional option to use the code for only VO and ignore mapping
  3. pair wise microbundle adjustment from the github code (possibly correct the mono depth estimation points as well) - https://github.com/Parskatt/micro-bundle-adjustment
  4. introduce tsdf fusion to get mesh - take code from Desktop/slam/rgbd_slam/kinfu_gui.py line 44
  '''



  '''
  i_pose = np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ) 

  local_maps = []
  sparse_poses = []
  all_coord_meshes = []

  frame_tuples = [[240, 250, 260, 270, 280, 290, 300, 310, 320, 330 ,
                   340, 350, 360, 370, 380, 390, 400, 410, 420, 430 , 
                   440, 450, 460, 470, 480, 490, 500, 510, 520, 530 ]]

  for f in frame_tuples:
    m = mapper(frame_numbers = f, init_pose = i_pose)
    m.run_sequence()
    #m.global_refinement()
    m.merge_map_poses()

    local_maps.append(m.scene_map)
    all_coord_meshes.extend(m.coord_meshes)
    sparse_poses.append(m.pose_sequence[-1])
    i_pose = np.dot(i_pose, m.pose_sequence[-1])


  print("showing final global map ")
  total_map = join_pcds(local_maps)
  total_map = total_map.voxel_down_sample(voxel_size=0.01)

  total_map_coords = [total_map]
  total_map_coords.extend(all_coord_meshes)
  project_rgbd.show_pcd(total_map_coords)

  print("Total number of points in the global scene ",np.array(total_map.points).shape[0])
  '''