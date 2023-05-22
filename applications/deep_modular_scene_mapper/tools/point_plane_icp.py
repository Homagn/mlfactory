#require latest open3d
#pip install -U pip>=20.3
#pip install -U open3d




import open3d as o3d
from . import project_rgbd
import numpy as np
import copy
import time


def create_pcd_from_points(points,colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def register_pair(pcd1, pcd2, viz = True):

  print("In pairwise registration")
  radius = 0.08*(30.0/255.0) #30.0

  
  div_scale = 10.0

  #bring the data to a similar scale as open3d datasets
  pt1 = np.asarray(pcd1.points)/1.0 #30.0
  cl1 = np.asarray(pcd1.colors)
  pt2 = np.asarray(pcd2.points)/1.0 #30.0
  cl2 = np.asarray(pcd2.colors)

  orig_pcds = [create_pcd_from_points(pt1,cl1), create_pcd_from_points(pt2,cl2)]

  source = orig_pcds[0]
  target = orig_pcds[1]
  

  #source = pcd1
  #target = pcd2

  print("0. Show the original pointclouds ")
  if viz:
    project_rgbd.show_pcd([source, target])

  current_transformation = np.identity(4)
  print("max of source ptcld ",np.max(np.asarray(source.points)))
  print("1-1. Downsample with a voxel size %.2f" % radius)
  source_down = source.voxel_down_sample(radius)
  target_down = target.voxel_down_sample(radius)

  print("1-2. Estimate normal.")
  source_down.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=60))
  target_down.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=60))

  print("showing normal estimated downsampled pointcloud ")
  if viz:
    project_rgbd.show_pcd([source_down, target_down])

  result_icp = o3d.pipelines.registration.registration_icp(
      source_down, target_down, radius, current_transformation,
      o3d.pipelines.registration.TransformationEstimationPointToPlane())
  print("result transformation ",result_icp.transformation)

  source = source.transform(result_icp.transformation)
  if viz:
    project_rgbd.show_pcd([source,target])

  return source, target, result_icp.transformation

if __name__ == '__main__':


  #partially aligned pointclouds using sift like feature matching + P3P registration
  #this gives much better results than using completely untransformed pointclouds because then icp between neighboring frames may get confused
  pcd_s = o3d.io.read_point_cloud("60_T.pcd")
  pcd_t = o3d.io.read_point_cloud("90_T.pcd")
  register_pair(pcd_s, pcd_t)