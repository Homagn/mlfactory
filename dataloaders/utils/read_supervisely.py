import numpy as np
import cv2
import json
import glob
import csv
import os

import copy
import math
from scipy.spatial import distance
#import params

import os
import sys
import shutil
os.environ['top'] = '../../'
sys.path.append(os.path.join(os.environ['top']))

from applications.lidar_post_detect import params


#this folder should contain the img and ann folders generated from within supervisely
#fname = '/datasets/lidar_post_detection/supervisely_annotated/extracted_lidar_data/' #if using supervisely
#fname = '/ml/misctools/tracking/tracking_annotations/' #if using home built tracker

#actual lidar person dataset
#fname = '/datasets/danfoss/lidar_person/supervisely_annotated/' #person detection from lidar
#generated lidar person dataset
#fname = '/datasets/danfoss/lidar_person/supervisely_fake/' #person detection from lidar
fname = params.annotated_data_folder

extract_name = None
#extract_name = '/datasets/danfoss/lidar_person/lidar_post_detection/extracted_lidar_data/' 

#this is the folder where supervisely annotations would be saved in our format to be accessed directly by network during training
#train_data_folder = '/datasets/danfoss/lidar_person/'
train_data_folder = params.train_folder


#make it true if you want to extract data from pcap files instead of reading the already extracted images in fname folder
read_from_extracted = False

image_ext = params.image_ext
#save_size = params.save_size

annotations = glob.glob(fname+'ann/'+'*.json')
count = 0

#maximum number of objects youre expecting to detect in the image
n_kp = params.num_pred_regions #do not change this to anything else for now because its hardcoded in the model sliding_window_conv.py take a look at comments in params.py
#n_kp = 10 


from_original_scale = True # True if the images in img folder are of the size 128x2048/ this is always true now

#current constraint - training_specific_image_resize[0]//training_specific_w_downscale must be =650

#image while reading from supervisely annotated img folder is first resized to this as the first preprocessing step
training_specific_image_resize = params.training_specific_image_resize

#training_specific_w_downscale = 2 #what is the ratio of original width of image/train_specific_image_resize width
#training_specific_h_downscale = 5 #what is the ratio of original height of image/train_specific_image_resize height

training_specific_w_downscale = params.training_specific_w_downscale #what is the ratio of original width of image/train_specific_image_resize width
training_specific_h_downscale = params.training_specific_w_downscale #what is the ratio of original height of image/train_specific_image_resize height


#training_specific_crop_w = [100,750]
#training_specific_crop_h = [0,64]

training_specific_crop_w = params.training_specific_crop_w
training_specific_crop_h = params.training_specific_crop_h

#ouster scans provide range (channel 0), intensity and reflectivity  information , if you want to select a grayscaled version of all put -1
channel_selection = params.channel_selection


#buffer to keep track of maximum object box sizes in the dataset
max_box_width = 0
max_box_height = 0





def writelog(fname, values):
    file1 = open(fname, 'a')
    writer = csv.writer(file1)
    fields1=values #is a list
    writer.writerow(fields1)
    file1.close()

def manhattan_arrange_keypoints(pts, axis=-1):
    #axis arguments tells it to focus only on a particular axis
    if axis==1:
        npts = np.array([ [32,p[1]] for p in pts  ])
    elif axis==0:
        npts = np.array([ [p[0],32] for p in pts  ])
    else:
        npts = copy.copy(np.array(pts))
    #print("got npts ",npts)


    pts = np.array(pts)
    
    ref_pt = np.array([[0,0]])
    distances = distance.cdist(npts, ref_pt, 'cityblock')
    dist_idcs = np.argsort(distances.flatten())
    
    #pts = pts[dist_idcs]
    #print("updated pts ",pts.flatten())
    return dist_idcs #returns [x1,y1,x2,y2,...]

def data_specific_kp_arrange(a_pts, a_pts_dims): #convert to consistent format for the data loader
    '''
    #say that number of regions the image is divided into is 18
    input of the model is 1 channel 64x650 image and the output is 5 channel 1x18 image
    so basically it divides the prediction over the width of the lidar image into 18 segments
    which means each segment covers roughly - 650/18= 36 pixels dividing the image uniformly from left to right

    for each segment it predicts (0 to 1) whether a pillar is present in the segment (1 if present)
    and also predicts (0 to 1) 0 if the pillar is in leftmost place of that segment and 1 if its present in the rightmost part of the segment, that is why 2 channel 
    and also predicts the center_y of the box
    and also predicts the width of the box
    and also predicts the height of the box

    values are arranged from the left most box to the right most box in the image
    '''
    #hor_div = 650//n_kp
    hor_div = (training_specific_crop_w[1]-training_specific_crop_w[0])//n_kp
    print("hor div ",hor_div)
    target = -1.0*np.ones((n_kp,1+4)) #if its -1 that means pillar is not there in that segment
    #1+4 because -- 1 because present/not present 0/1, +4 because of the 4 bounding box regression values
    
    for p in range(len(a_pts)) :
        
        y = a_pts[p][0]
        x = a_pts[p][1]
        w = a_pts_dims[p][0]
        h = a_pts_dims[p][1]

        idx = x//hor_div
        #print("idx ",idx)
        if idx>=n_kp:
            print("number of keypoints exceeded ")
            continue
        target[idx,:] = [1,float(y/training_specific_image_resize[1]),(x%hor_div)/hor_div,w,h]

    return list(target.flatten())


def get_center_x(x_val, w_crop = training_specific_crop_w, w_downscale = training_specific_w_downscale):
    x_v = x_val//w_downscale
    if x_v>=w_crop[0] and x_v<=w_crop[1]:
        x_v = x_v - w_crop[0]
        return x_v
    else: #annotation outside crop range
        return -1

def get_center_y(y_val, h_crop = training_specific_crop_h, h_downscale = training_specific_h_downscale):
    y_v = y_val//h_downscale
    if y_v>=h_crop[0] and y_v<=h_crop[1]:
        y_v = y_v - h_crop[0]
        return y_v
    else: #annotation outside crop range
        return -1

def degrade_vertical_resolution(im, every = 4):
    im_new = np.zeros((im.shape[0]//every, im.shape[1]))
    count = 0
    for row in range(0,im.shape[0],every): #take every 4th row
        im_new[count,:] = im[row,:]
        count+=1
    #size it back to original size
    im_new = cv2.resize(im_new,(im.shape[1],im.shape[0]))
    #im = median(im, disk(filter_kernel_size))
    return im_new

def convert_from_original_scale_image(image, crop = [training_specific_crop_h, training_specific_crop_w], w_resize = training_specific_image_resize, degrade_vres = False, channel_selection = channel_selection):
    
    image_range = image[:,:,0] #take only the signal channel out of stacked range/signal/ambient
    image_reflect = image[:,:,2]

    if channel_selection==0:
        image = image_range
    if channel_selection==-1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if degrade_vres:
        image = degrade_vertical_resolution(image)
    
    w = image.shape[1]
    image = cv2.resize(image,w_resize)
    print("image shape ",image.shape)
    image = image[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]


    return image



if __name__ == '__main__':

    #file = '/datasets/lidar_post_detection/annotated_lidar_data/keypoints.csv'
    file = train_data_folder+'keypoints.csv'
    csv_file = train_data_folder+'keypoints.csv'
    #images_file = '/datasets/lidar_post_detection/annotated_lidar_data/train/'
    images_file = train_data_folder+'train/'

    if(os.path.exists(file) and os.path.isfile(file)):
      os.remove(file)
      print("previous label file deleted")
      print("creating a new file")
    else:
      print("file not found, creating a new one")

    if(os.path.isdir(images_file)):
      #os.rmdir(images_file)
      print("Folder already exists ")
      shutil.rmtree(images_file)
      os.mkdir(images_file)
    else:
      print("file not found, creating a new one")
      os.mkdir(images_file)




    for n in annotations:
        f = open(n,)
        
        #im_name = n[n.find("\\")+1:n.find(image_ext)]
        im_name = n[n.rfind("/")+1:n.find(image_ext)]

        print("file name ",n)
        print("image name ",im_name)

        data = json.load(f)
        if data['objects']!=[]:
            #print("got data ",data)
            c_pts = []
            box_dims = []

            '''
            if data['objects']['geometryType']=='point': #for keypoint based annotation
                for point in data['objects']:
                    for p in point['points']['exterior']:
                        if from_original_scale:
                            pts.append([32,p[0]]) #0 is the x value 1 is the y value, assuming we are detecting the post center in the center of the entire height of the image
            '''
            
            if data['objects'][0]['geometryType']=='rectangle': #for bounding box type annotation
                #for point in data['objects']:
                for point in  data['objects']:
                    top_left_x = point['points']['exterior'][0][0]
                    bottom_right_x = point['points']['exterior'][1][0]

                    top_left_y = point['points']['exterior'][0][1]
                    bottom_right_y = point['points']['exterior'][1][1]

                    if from_original_scale:
                        #both image width and height are downscaled by 2 while being input to model
                        x_v = get_center_x( (bottom_right_x+top_left_x)//2   )
                        y_v = get_center_y( (bottom_right_y+top_left_y)//2   )
                        if x_v!=-1:
                            c_pts.append([y_v,  x_v  ]) #0 is the x value 1 is the y value, assuming we are detecting the post center in the center of the entire height of the image
                            
                            width = math.fabs(bottom_right_x-top_left_x)/training_specific_w_downscale
                            height = math.fabs(bottom_right_y-top_left_y)/training_specific_h_downscale

                            if width>max_box_width:
                                max_box_width = width
                            if height>max_box_height:
                                max_box_height = height

                            #box_dims.append([width,height])
                            #normalizations
                            box_dims.append([width/params.max_bb_dim,height/params.max_bb_dim])


            
            if c_pts==[]:
                continue #skip adding the annotations to the training dataset
            


            #print("got points ",pts)

            distances = manhattan_arrange_keypoints(c_pts, axis=1) #typically a clockwise pattern - bottom left, top left, top right, bottom right
            
            a_pts = np.array(c_pts)[distances]
            a_pts_dims = np.array(box_dims)[distances]

            print("count ",count)
            print("got manhattan arranged points ",a_pts)
            
            #write the image regression labels
            #values = ["Image number ",im_name,a_pts[1],a_pts[2],a_pts[3],a_pts[4],a_pts[5],a_pts[7]]
            values = ["Image number ",count]
            #values.extend(params.data_specific_kp_arrange(a_pts))
            values.extend(data_specific_kp_arrange(a_pts, a_pts_dims))

            print("got arranged points ",values)
            #writelog('data/keypoints.csv',values)
            writelog(csv_file,values)
            
            #write the image in a convenient name, size and folder location
            if read_from_extracted:
                print("reading image ",extract_name+im_name+'.png')
                image = cv2.imread(extract_name+im_name+'.png')
            else:
                print("reading image ",fname+'img/'+im_name+'.png')
                image = cv2.imread(fname+'img/'+im_name+'.png')
            
            image = convert_from_original_scale_image(image)
            #image_s = cv2.resize(image, save_size)
            
            #cv2.imwrite('data/train/'+str(count)+'.png',image)
            cv2.imwrite(images_file+str(count)+'.png',image)


            count+=1
        # Closing file
        f.close()

    print("Total number of annotated images ",count)
    print("calculated maximum box width ",max_box_width)
    print("calculated maximum box height ",max_box_height)
    print("run the code again by changing max_bb_dim variable in params.py")