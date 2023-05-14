import numpy as np
import math
import cv2
import copy

import os
import sys
os.environ['top'] = '../'
sys.path.append(os.path.join(os.environ['top']))
from applications.lidar_post_detect import params

#maximum number of objects youre expecting to detect in the image
n_kp = params.num_pred_regions ##do not change this to anything else for now because its hardcoded in the model sliding_window_conv.py
#n_kp = 10
training_image_height = params.model_input_image_height

#maximum value for any bounding box width or height
#max_bb_dim = 46
#max_bb_dim = 135
max_bb_dim = params.max_bb_dim


#anno_data_dir = "/datasets/lidar_post_detection/"
#anno_data__sub_dir = "annotated_lidar_data/"

#anno_data_dir = "/datasets/danfoss/lidar_person/"
anno_data_dir = params.train_folder
anno_data__sub_dir = "" #if any subdirectory contain the train folder


def training_style_specific_convert_Y(reg_list):
    #view the idea.txt or readme to get an idea on how this specific application is trained
    y = np.zeros((n_kp,5))
    
    y_flat = reg_list

    for i in range(len(y_flat)):
        if y_flat[i][0]==-1.0: #means feature is not present
            y[i,:] = [0.0, 0.0, 0.0, 0.0, 0.0] #0.0 means object is not present, so regression loss wont be considered
        else:
            y[i,:] = y_flat[i] #1.0 means the object is present, so loss will be based on regression and classification
    #print("y for model ",y)
    return y

def training_style_specific_convert_X(image):
    #view the idea.txt or readme to get an idea on how this specific application is trained
    image = np.array(image/255.0, dtype = np.float32)
    return image

'''
def check_remove_faint_feature(x,y,resize_height = 256):
    #in lidar data sometimes features are very faint because of distance
    #these are valid yes but still model should ignore them
    #otherwise training may be bad
    #its not yet perfect dont use it

    inp = x
    inp = cv2.resize(inp,(inp.shape[1],resize_height))
    #print("max and min of input ",np.max(inp), np.min(inp)) #checked between 0 and 1

    outp = y
    y_mod = copy.copy(outp)

    window_len = inp.shape[1]//n_kp
    #print("window len ",window_len)
    for a in range(outp.shape[0]):
        feature_present = outp[a,0]
        #print("feature_present ",feature_present)
        reg_loc = outp[a,1]

        if feature_present>0.5:
            #Draw the keypoint on the input image
            center_coordinates = (window_len*a+int(window_len*reg_loc), 70)
            interest_patch = x[center_coordinates[1]-60:center_coordinates[1]+50, center_coordinates[0]-5:center_coordinates[0]+5]
            print("mean of the selected region ",np.mean(interest_patch))
            if(np.mean(interest_patch))<100.0:
                y[a,:] = [0.0, 0.0]
'''



def probe_model_input_output(intput_tensor,output_tensor, image_label, resize_height = 256, confidence_thresh = 0.5, moving_average_pred = None):
    #just before tensors are feed into model
    #back convert them and check for correct range and correct correspondence
    
    #print("input tensor shape ",intput_tensor[0].shape)
    #print("output tensor shape ",output_tensor[0].shape)

    


    inp = np.array(intput_tensor[0])
    inp = cv2.resize(inp,(inp.shape[1],resize_height))
    inp = np.stack([inp,inp,inp],axis=2) #make gray to RGB
    #print("max and min of input ",np.max(inp), np.min(inp)) #checked between 0 and 1

    outp = np.array(output_tensor[0])

    #when confidence thresh of 1 is passed then only max confidence boxes are shown
    #out of all the object detections in the frame get the detection for which model has highest confidence
    feature_present_scores = [outp[a,0] for a in range(outp.shape[0])]
    max_confidence = np.max(feature_present_scores)
    if confidence_thresh==1.0:
        confidence_thresh = max_confidence
    


    model_input_height = inp.shape[0]
    resize_h = resize_height/training_image_height

    window_len = inp.shape[1]//n_kp
    #print("window len ",window_len)
    for a in range(outp.shape[0]):
        feature_present = outp[a,0]
        #print("feature_present ",feature_present)
        reg_loc_y = outp[a,1]*training_image_height
        reg_loc_x = outp[a,2]
        bb_w = outp[a,3]
        bb_h = outp[a,4]

        '''
        print("feature present ",feature_present)
        print("reg_loc_y ",reg_loc_y)
        print("reg_loc_x ",reg_loc_x)
        print("bounding box width ",bb_w)
        print("bounding box height ",bb_h)
        print("resize_h ",resize_h)
        '''

        if feature_present>=confidence_thresh:
            #Draw the keypoint on the input image
            center_coordinates = (window_len*a+int(window_len*reg_loc_x), int(reg_loc_y*resize_h))
            #print("center_coordinates ",center_coordinates)
            color = (255, 0, 0)

            dw = int((bb_w*max_bb_dim)//2)
            dh = int((bb_h*max_bb_dim*resize_h)//2)
            #print("dw dh ",dw,dh)

            start_point = (center_coordinates[0]-dw, center_coordinates[1]-dh)
            end_point = (center_coordinates[0]+dw, center_coordinates[1]+dh)
            thickness = 2

            

            
            #print("started going through dictionary ")
            if moving_average_pred!=None: #means combine current prediction with past predictions to achieve smooth preds, very useful in object tracking in videos
                #basically a dictionary is passed if its not none
                center_hash = str(center_coordinates[0])+"_"+str(center_coordinates[1])
                center_assigned = False

                for k in moving_average_pred.keys():
                    if k=="variance_capture" or k=="moving_window_size":
                        continue
                    
                    query_center = [int(k[:k.find("_")]), int(k[k.find("_")+1:]) ]
                    #print("query center ",query_center)
                    #print("center_coordinates ",center_coordinates)

                    if math.fabs(center_coordinates[0]-query_center[0])< moving_average_pred["variance_capture"][0] and math.fabs(center_coordinates[1]-query_center[1])< moving_average_pred["variance_capture"][1]:
                        #this means a prev prediction box wth roughly the same center as current pred has been located
                        moving_average_pred[k].append( np.array([start_point[0], start_point[1], end_point[0], end_point[1] ] ) )

                        new_pts = list(np.mean(moving_average_pred[k], axis=0))
                        new_pts = [int(p) for p in new_pts]
                        
                        start_point = ( new_pts[0], new_pts[1] )
                        end_point = ( new_pts[2], new_pts[3] )

                        inp = cv2.rectangle(inp, start_point, end_point, color, thickness)
                        center_assigned = True
                        #print("center_assigned")

                    if len(moving_average_pred[k]) > moving_average_pred["moving_window_size"]:
                        moving_average_pred[k] = []

                if not center_assigned:
                    #print("center not assigned")
                    moving_average_pred[center_hash] = [ np.array([start_point[0], start_point[1], end_point[0], end_point[1] ] ) ]
                    inp = cv2.rectangle(inp, start_point, end_point, color, thickness)


            else:
                inp = cv2.rectangle(inp, start_point, end_point, color, thickness)




    if params.crop_pred_viz!=[]:
        inp = inp[params.crop_pred_viz[1][0]:params.crop_pred_viz[1][1], params.crop_pred_viz[0][0]:params.crop_pred_viz[0][1]] 

    cv2.imshow(image_label, inp)
    cv2.waitKey(100)
    return moving_average_pred
    


#use this loader for convolution regression frameworks
#first images are keypoint annotated in supervisely
#then download json+images from the website
#then run read_supervisely.py which organizes the jsons and images into a csv file+training images
#this class below loads the csv file labels and corresponding images
class dataloader(object): 
    def __init__(self):
        self.init_labels()
        self.batch_size = 16
    
    def init_labels(self):
        self.ytrain_raw = np.loadtxt(anno_data_dir+anno_data__sub_dir+"keypoints.csv", delimiter=',', skiprows=0, usecols=range(1,(n_kp*5)+2))
        #print("got y train raw ",self.ytrain_raw)

        self.idcs = np.unique(self.ytrain_raw[:,0])
        #print("got unique image indices ",self.idcs)


        self.y_train = {}

        #print("self ytrain raw ",self.ytrain_raw[0,:], len(self.ytrain_raw[0,:]))
        

        for idx in range(self.ytrain_raw.shape[0]):
            self.y_train[int(self.ytrain_raw[idx,0])] = []
            #every 5 numbers store the following
            #--------
            #is there a feature ? 0/1
            #whats the center x of the feature ?
            #center y
            #width
            #height
            #--------

            for i in range(1,n_kp*5,5):
                elems = self.ytrain_raw[idx,i:i+5]
                #elem_scaled = (elem-self.scales[i][1])/(self.scales[i][0]-self.scales[i][1])
                self.y_train[int(self.ytrain_raw[idx,0])].append(elems)
                #print("elems ",elems)
            #sys.exit(0)


    def sample_one(self, idx = 0):
        if idx==-1:
            id1 = np.random.choice(self.idcs, size=1)
        else:
            id1 = self.idcs[idx]

        #print("got randomly sampled id ",id1)

        x = cv2.imread(anno_data_dir+anno_data__sub_dir+'train/'+str(int(id1))+'.png', cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('x',x)
        #cv2.waitKey(0)



        y_flat = self.y_train[int(id1)]
        #print("got y ",y_flat)

        
        return x,y_flat

    def sample_batch(self, x_conversion_func = training_style_specific_convert_X, y_conversion_func = training_style_specific_convert_Y, bsize=16):
        xb = []
        yb = []
        bsize = self.batch_size

        for _ in range(bsize):
            x,y = self.sample_one(idx=-1)
            y = y_conversion_func(y) #this y can now be directly converted to tensor to feed into model target
            x = x_conversion_func(x) #this x can now be directly converted to tensor to feed into model input
            xb.append(x)
            yb.append(y)

        xb = np.stack(xb, axis=0)
        yb = np.stack(yb, axis=0)

        print("xb shape ",xb.shape)
        print("yb shape ",yb.shape)
        return xb, yb



if __name__ == '__main__':
    l = dataloader()
    
    x,y = l.sample_one(-1) #-1 for sampling a random one from the data
    cv2.imshow('raw image',x)
    cv2.waitKey(0)
    y = training_style_specific_convert_Y(y)
    print("Labels y for the image ",y)
    

    probe_model_input_output([x],[y], "Image with annotation")
    cv2.waitKey(0)

    #check_remove_faint_feature(x,y)
    #probe_model_input_output([x],[y], "Image with annotation faint features removed")

    
    l.sample_batch(training_style_specific_convert_X, training_style_specific_convert_Y)
    cv2.waitKey(0)