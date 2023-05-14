#utilize the CIHP dataset to extract the contour shape of human
#randomly distort and reposition the contours to obtain a human contours image
#apply for transfer learning to lidar object detection (during inference lidar image passed through canny filter)


import os
import sys

import numpy as np
import cv2

from glob import glob

import shutil

os.environ['top'] = '../../'
sys.path.append(os.path.join(os.environ['top']))
from datacreators.utils import cv_annotator


num_random_overlays = 5
data_dir = "/datasets/danfoss/lidar_person/"
fake_data_dir = "/datasets/danfoss/lidar_person/supervisely_fake/"
num_generate = 5000 #generate 40000 fake images

def read_files():
    path = "/datasets/CIHP/human_part_seg/Training"
    NUM_TRAIN_IMAGES = 30000
    train_images = sorted(glob(os.path.join(path, "Images/*")))[:NUM_TRAIN_IMAGES]
    train_masks = sorted(glob(os.path.join(path, "Human/*")))[:NUM_TRAIN_IMAGES]

    data = {"image":train_images, "masks":train_masks}

    return data, len(data["image"])


def get_contour(mask, fill = False):
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)    
    image_copy = np.zeros_like(mask)
    if fill:
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=cv2.FILLED)
    else:
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    #cv2.imshow('None approximation', image_copy)
    #cv2.waitKey(0)
    return image_copy, contours

def contour_bbox(cont_img):
    x_c,y_c,w,h = cv2.boundingRect(cont_img)
    #cv2.rectangle(cont_img,(x_c,y_c),(x_c+w,y_c+h),(255,255,255),2)
    #cv2.imshow("bounding box contour ",cont_img)
    #cv2.waitKey(0)
    return x_c,y_c,w,h

def overlay_contour_background(cont_img, back_img):
    #back = cv2.imread(back_img,0)
    back = back_img
    H = back.shape[0]
    W = back.shape[1]

    if cont_img.shape[0]>H or cont_img.shape[1]>W:
        cont_img = cv2.resize(cont_img, ( H//2 , H//2))

    random_shrink = np.random.randint(5,10)/10.0
    random_widen = 1+np.random.randint(6,9)/10.0
    h = int(H*random_shrink)

    if h>cont_img.shape[0]:
        h = cont_img.shape[0]

    

    

    overlay = cv2.resize(cont_img, ( cont_img.shape[1]//(int(cont_img.shape[0]//h)) , h))
    #print("overlay shape ",overlay.shape)

    '''
    cv2.imshow("overlay ",overlay*255.0)
    cv2.waitKey(0)
    '''

    #widen the contour image to create a lidar liek effect
    overlay = cv2.resize(overlay, ( int(overlay.shape[1]*random_widen), overlay.shape[0]))

    '''
    cv2.imshow("widened overlay ",overlay*255.0)
    cv2.waitKey(0)
    '''

    cont, cont_pts = get_contour(overlay)

    '''
    cv2.imshow("small overlay contour ",cont)
    cv2.waitKey(0)
    '''

    start_x = np.random.randint(0, W- cont.shape[1])
    start_y = np.random.randint(0, H- cont.shape[0])

    #first add the background feature on top of the contour
    cont+= back[start_y:start_y+cont.shape[0], start_x:start_x+cont.shape[1]]
    #redraw the contour with increased thickness
    cv2.drawContours(image=cont, contours=cont_pts, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    #in case stuff from background entered inside the contour, fill the inside with 0s again
    cv2.drawContours(image=cont, contours=cont_pts, contourIdx=-1, color=(0, 0, 0), thickness=cv2.FILLED)
    #reassign the modified part of the image to the entire image
    back[start_y:start_y+cont.shape[0], start_x:start_x+cont.shape[1]] = cont


    #draw the bounding box rectangle just to check
    #cv2.rectangle(back,(start_x,start_y),(start_x+cont.shape[1],start_y+cont.shape[0]),(255,255,255),2)
    annot_box = [start_y,start_x,start_y+cont.shape[0], start_x+cont.shape[1]]


    '''
    cv2.imshow("final overlaid ",back)
    cv2.waitKey(0)
    '''

    return back, annot_box




def load(image_path, mask_path):
    x = cv2.imread(image_path)
    y = cv2.imread(mask_path,0) #force read grayscale
    return x, y

def extract_overlay_contours(y):
    u = np.unique(y)
    #print("uniques ",u)
    ru = np.random.choice(u[1:])
    y[y!=ru] = 0
    y[y==ru] = 1

    cont, _ = get_contour(y)
    x_c,y_c,w,h = contour_bbox(cont)
    single_person_contour = y[int(y_c):int(y_c+h),int(x_c):int(x_c+w)]

    return single_person_contour


    


def sample_one( data, datalen, sample_number = -1):
    #print("number of files in dataset ",self.datalen)
    if sample_number==-1:
        sample_number = np.random.randint(0, datalen)
    
    #print("viewing sample number ",sample_number)
    background = cv2.imread("contours.png",0)

    annotation_boxes = []
    overlays = np.random.randint(1,num_random_overlays)

    for i in range(overlays):
        sample_number = np.random.randint(0, datalen)
        rgb, mask= load(data["image"][sample_number], data["masks"][sample_number])
        single_person_contour =  extract_overlay_contours(mask)

        
        o1, abox = overlay_contour_background(single_person_contour, background)
        background = o1
        annotation_boxes.append(abox)

    '''
    #checked okay
    cv2.imshow("final overlaid ",background)
    cv2.waitKey(0)
    print("got annotation boxes ",annotation_boxes)
    '''

    return background, annotation_boxes


def save_img_ann(img, ann, idx):
    #print("saving image and annotation ")
    #idx = 0

    file_name = fake_data_dir+"ann/"+str(idx)+".png.json"
    d = cv_annotator.load_annotation_dict(file_name)
    for object_box in ann:
        cv_annotator.populate_save_annotation_dict(file_name, d,object_box)
    imfilesave = fake_data_dir+"img/"+str(idx)+".png"

    cv2.imwrite(imfilesave, img)
    #print("saved image file name  ",imfilesave)




data, datalen = read_files()

print("Removing pre-existing directory ...")
try:
    location = data_dir
    path = os.path.join(location, "supervisely_fake")
    # removing directory
    shutil.rmtree(path)
except:
    print("Directory didnt exist earlier")

print("recreating directory ...")
os.mkdir(fake_data_dir)
os.mkdir(fake_data_dir+"ann/")
os.mkdir(fake_data_dir+"img/")


for n in range(num_generate):
    fake_img, fake_ann = sample_one(data, datalen)
    save_img_ann(fake_img, fake_ann, n)
    print("Generated ",n, end='\r')

