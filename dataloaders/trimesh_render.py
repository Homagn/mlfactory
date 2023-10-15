# This dataloader contains function to read a 3d object model (eg collada/obj files)
# render it using pyrender and trimesh
# return color and depth maps of the objects at various angles




import sys, os
# An installation agnostic method to find and link to root of the package which is mlfactory
#==========================================================
import re
try: #testing the functions locally without pip install
  import __init__
  cimportpath = os.path.abspath(__init__.__file__)
  if 'extensions' in cimportpath:
    print("local testing ")
    import mlfactory
    cimportpath = os.path.abspath(mlfactory.__file__)

except: #testing while mlfactory is installed using pip
  print("Non local testing")
  import mlfactory
  cimportpath = os.path.abspath(mlfactory.__file__)

main_package_loc = cimportpath[:cimportpath.rfind('mlfactory')+len('mlfactory')]
print("got main package location ",main_package_loc)


os.environ['top'] = main_package_loc
sys.path.append(os.path.join(os.environ['top']))
#==========================================================

import math
from math import cos, sin, radians
from visualizers import project_rgbd
import numpy as np
import trimesh
import pyrender #pip install pyrender, pip install pycollada
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

import copy

#used by the sampler function
import torch





# default variables
check_reprojection = False
pcds = []
offscreen_width = 640
offscreen_height = 480
pitch_variations = 4
yaw_variations = 20


zipped_models_location = "/datasets/shapenet/car_models" #contains zip files which contain model files in dae
extracted_models_location = "/datasets/shapenet/extracted_cars/"
# in this location each 3D model will be rendered at multiple views 
render_models_locations = "/datasets/shapenet/render_cars/"







def trig(angle):
  r = radians(angle)
  return cos(r), sin(r)


def get_transform(rotation, translation):
  xC, xS = trig(rotation[0])
  yC, yS = trig(rotation[1])
  zC, zS = trig(rotation[2])
  dX = translation[0]
  dY = translation[1]
  dZ = translation[2]
  Translate_matrix = np.array([[1, 0, 0, dX],
                               [0, 1, 0, dY],
                               [0, 0, 1, dZ],
                               [0, 0, 0, 1]])
  Rotate_X_matrix = np.array([[1, 0, 0, 0],
                              [0, xC, -xS, 0],
                              [0, xS, xC, 0],
                              [0, 0, 0, 1]])
  Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                              [0, 1, 0, 0],
                              [-yS, 0, yC, 0],
                              [0, 0, 0, 1]])
  Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                              [zS, zC, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
  return np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix)))


def spiral_camera_poses(num_elevations = 4, num_rotations = 20):
  start_pitch = -20
  start_yaw = 0

  pres = 60.0/num_elevations
  qres = 360.0/num_rotations

  transforms = []
  rpy = []


  for p in range(num_elevations): #try 10 random camera rotations
    for q in range(num_rotations):

      roll = 0.0
      pitch = start_pitch + pres*p
      yaw = start_yaw + qres*q


      T = get_transform([0,pitch,yaw], [0,0,0] )
      transforms.append(T)
      rpy.append([0, pitch, yaw])

  return transforms, rpy




def normalize_bounds(mesh):
  rescale = max(mesh.extents)/2.
  tform = [
    -(mesh.bounds[1][i] + mesh.bounds[0][i])/2.
    for i in range(3)
  ]
  matrix = np.eye(4)
  matrix[:3, 3] = tform
  mesh.apply_transform(matrix)
  matrix = np.eye(4)
  matrix[:3, :3] /= rescale
  mesh.apply_transform(matrix)
  return mesh


def prepare_data_folder(zip_location, target_location = "/datasets/shapenet/extracted_cars/"):
  # zip location is the name of the folder that contains all the zip files 
  # each zip file should store the 3d model of the object as model.dae file
  # this function should retrieve all those models from the zip files and store it in target_location within interger numbered folders                                                          
  import glob      
  from zipfile import ZipFile

  os.chdir(zip_location)

  count = 0
  for file in list(glob.glob('*.zip')):                                       
    print("name ",file)              # <-- Things done for each file
    with ZipFile(file, 'r') as zObject:
      os.chdir(target_location)
      #zObject.printdir()
      os.mkdir(str(count))
      zObject.extractall(str(count)+"/")
      os.chdir(zip_location)
    count+=1


def check_pointcloud(color, depth):
  # reproject the depth and color maps and check the pointcloud
  print("Now trying to show the projected pointcloud ")
  #depth = (depth-np.min(depth))/np.max(depth) 
  depth = depth/3.0
  depth = depth*255

  camera_params = {
                  "fx": 350.0,
                  "fy": 350.0,
                  "centerX": color.shape[0]//2,
                  "centerY": color.shape[1]//2,
                  "scalingFactor": 224
                  }
  pcd = project_rgbd.pcd_from_rgbd_native(color, depth, camera_params)
  project_rgbd.show_pcd([pcd])










def create_multiview_dataset():

  generation_poses, rpy = spiral_camera_poses(num_elevations = pitch_variations, num_rotations = yaw_variations)
  #initialize scene
  scene = pyrender.Scene()

  for model_num in range(97):
    random_model_load = extracted_models_location+str(model_num)+"/model.dae"
    load_trimesh = trimesh.load(random_model_load, force = 'mesh')


    #scale the loaded mesh to lie within a unit cube
    load_trimesh = normalize_bounds(load_trimesh)
    mesh = pyrender.Mesh.from_trimesh(load_trimesh)
    print("mesh scale ",mesh.scale)


    #set default model pose for a particular model
    model_pose = np.eye(4)
    model_pose[0,3] = 0.1
    model_pose[2,3] = -np.min(load_trimesh.vertices[:,2])

    for pose in range(len(generation_poses)): #try 10 random camera rotations
      g = generation_poses[pose]
      #add the model
      scene.add(mesh, pose = model_pose)

      #view the scene without offscreen render (for check only)
      #pyrender.Viewer(scene, use_raymond_lighting=True)



      #initialize perspective camera and light
      camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
      #camera = pyrender.OrthographicCamera(1.0,1.0) # bad view
      light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,innerConeAngle=np.pi/16.0,outerConeAngle=np.pi/6.0)
      #set the viewing perspective from the camera
      s = np.sqrt(2)/2 #sets the 45 degree camera angle
      camera_pose = np.array([
          [0.0, -s,   s,   1.9],
          [1.0,  0.0, 0.0, 0.0],
          [0.0,  s,   s,   1.7],
          [0.0,  0.0, 0.0, 1.0],
       ])
      camera_pose = np.matmul(g, camera_pose)
      # add light and camera to the scene
      scene.add(camera, pose=camera_pose)
      scene.add(light, pose=camera_pose)
      # render the scene at the particular camera angle and get the color and depth maps
      r = pyrender.OffscreenRenderer(viewport_width=offscreen_width, viewport_height=offscreen_height)
      color, depth = r.render(scene)
      r.delete()





      # show stats and show the depth and color images
      print("max of depth ",np.max(depth), np.mean(depth))
      cv2.imshow("color ",color)
      #cv2.imshow("depth ",depth/3.0 ) #using standard max = 3.0 for all the depth maps for the specific model (need to be found out automatically somehow)
      cv2.waitKey(1)


      if os.path.exists(render_models_locations+str(model_num))==False:
        print("folder does not exist, creating one ")
        os.mkdir(render_models_locations+str(model_num))

      ele = pose//20
      yaw = pose%20
      cv2.imwrite(render_models_locations+str(model_num)+"/"+str(ele)+"_"+str(yaw)+".png", color)





      # need to implement proper depth map fusion so that all the different view depth maps can be merged into a single pcd
      if check_reprojection:
        check_pointcloud(color, depth)


      scene.clear() #remove everything


def normalize_in_range(img):
  img = img/255.0 #img -> [0,1]
  img = (img*2.0)-np.ones_like(img) # = [-1,1] typical range for diffusion models
  return img


def sample():
  random_model = np.random.randint(1, 97)

  random_ele = np.random.randint(4)
  random_yaw = np.random.randint(20)

  random_ele_ref = np.random.randint(4)
  random_yaw_ref = np.random.randint(20)

  #read a grayscale and later resize
  img_target = cv2.imread(render_models_locations+str(random_model)+"/"+str(random_ele)+"_"+str(random_yaw)+".png")
  img_ref = cv2.imread(render_models_locations+str(random_model)+"/"+str(random_ele_ref)+"_"+str(random_yaw_ref)+".png")

  img_target = cv2.resize(img_target, (64,64))
  img_ref = cv2.resize(img_ref, (64,64))

  ele_diff = random_ele - random_ele_ref
  yaw_diff = random_yaw - random_yaw_ref

  

  diff_label = (ele_diff+3)*19 + (yaw_diff+19)

  #checks
  '''
  print("ele diff ",ele_diff)
  print("yaw diff ",yaw_diff)
  print("diff label ",diff_label) #max label = 6*19 + (19+19) = 152 labels (pose difference variations)
  print("image maxes ",np.max(img_target), np.max(img_ref) )

  cv2.imshow("image target",img_target)
  cv2.imshow("image ref ",img_ref)
  cv2.waitKey(0)
  '''
  
  
  

  xt = normalize_in_range( img_target )
  xr = normalize_in_range(img_ref)

  #print("max and min after normalizing ",np.max(xt), np.min(xt))
  #print("max and min after normalizing ",np.max(xr), np.min(xr))


  l = diff_label

  return xt, xr, l

def sample_batch(sz=2):
  xtb = []
  xrb = []
  L = []

  for _ in range(sz):
    xt, xr, l = sample()
    xtb.append(xt)
    xrb.append(xr)
    L.append(l)

  XT = np.stack(xtb).astype(np.float32)
  XR = np.stack(xrb).astype(np.float32)
  L = np.stack(L, axis=0)
  #print("numpy shapes ",XR.shape, XT.shape, L.shape)

  #XT = torch.from_numpy(XT).view((sz, 3, 64, 64)) #.to(device)
  #XR = torch.from_numpy(XR).view((sz, 3, 64, 64))

  XT = torch.from_numpy(XT).permute(0,3,1,2) #.to(device)
  XR = torch.from_numpy(XR).permute(0,3,1,2)

  L = torch.tensor(L, dtype=torch.long)#.to(device)

  #print("maxes ",torch.max(XT), torch.min(XR))

  #print("torch shapes ",XT.shape, XR.shape, L.shape)

  return XT, XR, L





if __name__ == '__main__':
  
  #extract zip files containing models
  '''
  if os.path.exists(extracted_models_location):
    print("target extracted models location ",extracted_models_location)
    print("either model zip files has already been extracted or location has not been created ")
  else:
    prepare_data_folder("/datasets/shapenet/car_models", target_location = "/datasets/shapenet/extracted_cars/")
  '''

  # create multiple view dataset consisting of images taken for each model at multiple camera angles
  # used for training novel view synthesis
  #create_multiview_dataset()

  #sample()
  sample_batch()




