
import sys,os

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

from applications.multiview_sba_recon import sfm
import numpy as np

if __name__ == '__main__':
  img_dir = "/datasets/reconstruction_examples/GustavIIAdolf/"
  iparams = sfm.get_image_params(img_dir)
  images = sorted( filter( lambda x: os.path.isfile(os.path.join(img_dir, x)), os.listdir(img_dir) ) )
  
  cameras = []
  point_cloud = []
  point_color = []
  
  #Gustav
  K = np.array([[ 2.39395217e+03 ,-3.41060513e-13,  9.32382177e+02],
               [ 0.00000000e+00,  2.39811854e+03,  6.28264995e+02],
               [ 0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])
  K = np.array(K, dtype=float)
  

  sfm.register_images(img_dir, images, iparams, K, cameras, point_cloud, point_color )