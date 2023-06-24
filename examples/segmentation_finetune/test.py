import os,sys
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

from datacreators.nn_seg import segmenter




if __name__ == '__main__':
    ino = 33 #48 #44
    s = segmenter(loadpath = '/datasets/pretrained_weights/pillar_detect/weights.pt')
    #s.run_predictor(f'./food_packet/Images/{ino:03d}.jpg')
    s.run_predictor(f'/datasets/small_segmentation/lidar_pillars/Images/'+"033"+'.jpg')
    boxes = s.extract_boxes()
    print("got bounding box annotations ",boxes)

    #x,y,w,h = s.extract_roi(savefilename = "output.png")




