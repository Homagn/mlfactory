import os
import sys
#import bottleneck as bn
from ouster import client, pcap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import hsv_to_rgb

import cv2
import copy
import shutil
from datetime import datetime as dt
from sklearn.cluster import KMeans
import open3d as o3d

cimportpath = os.getcwd()
if cimportpath[cimportpath.rfind("/")+1:]=="dataloaders": #if this module is called from dataloaders code
    os.environ['top'] = '../'
    os.environ['applications'] = '../visualizers'
    sys.path.append(os.path.join(os.environ['top']))
    sys.path.append(os.path.join(os.environ['applications']))

from visualizers import project_rgbd
from datacreators.utils import cv_annotator

'''

'''

#================================
#all different types of normalizations

def get_max_ignore_outliers(arr):
    arr_flat = arr.flatten()
    top_n_idx = int(len(arr_flat)*0.0005)
    #return np.median(-bn.partition(-arr_flat, top_n_idx)[:top_n_idx])
    return np.median(-np.partition(-arr_flat, top_n_idx)[:top_n_idx])

def ring_norm(range_rs):
    #print(np.max(range_rs))
    #return range_rs/90000.0
    
    #my special normalization produces nice contrast on actual lidar data
    range_rs = np.nan_to_num(range_rs)
    range_rs[range_rs==0]=1 #the nan values that got converted to 0, assume they are very close points

    col_max = range_rs.max(axis=1)
    range_rs = range_rs/col_max[:,np.newaxis]

    ring_max = range_rs.max(axis=0)
    ring_norm = range_rs/ring_max[np.newaxis,:]
    #full_norm = (ring_norm-np.mean(ring_norm))/(np.max(ring_norm)-np.min(ring_norm))
    full_norm = (ring_norm)/np.max(ring_norm)
    return full_norm
    

def inverse_norm(range_rs):
    #inverse normalization (closer is whiter)
    #used this in gazebo simulations
    inv_thresh = 500.0
    range_rs = np.nan_to_num(range_rs)
    range_rs[range_rs==0]=(1/inv_thresh)
    inv_norm = inv_thresh/range_rs
    inv_norm[inv_norm>1] = 1.0
    full_norm = inv_norm 
    return full_norm

def easy_norm(range_rs, chan_num):
    print("max value ",np.max(range_rs))
    if normalize_value=='max':
        div = np.max(range_rs)
    else:
        div = normalize_value[chan_num] #15000 (range), 500 (ambient)
    result = range_rs/div
    result[result>div]=div
    return result

#ouster_normalize = easy_norm

def ouster_normalize(data: np.ndarray, channel_num: int = 0, percentile: float = 0.05):
    """Normalize and clamp data for better color mapping.
    This is a utility function used ONLY for the purpose of 2D image
    visualization. The resulting values are not fully reversible because the
    final clipping step discards values outside of [0, 1].
    Args:
        data: array of data to be transformed for visualization
        percentile: values in the bottom/top percentile are clambed to 0 and 1
    Returns:
        An array of doubles with the same shape as ``image`` with values
        normalized to the range [0, 1].
    """
    min_val = np.percentile(data, 100 * percentile)
    max_val = np.percentile(data, 100 * (1 - percentile))
    # to protect from division by zero
    spread = max(max_val - min_val, 1)
    field_res = (data.astype(np.float64) - min_val) / spread
    return field_res.clip(0, 1.0)


#==============================================


def load_channel(lidar_channel, scan):
    if lidar_channel=='range':
        range_field = scan.field(client.ChanField.RANGE)
    if lidar_channel=='near_ir':
        range_field = scan.field(client.ChanField.NEAR_IR)
    if lidar_channel=='signal':
        range_field = scan.field(client.ChanField.SIGNAL)
    if lidar_channel=='reflect':
        range_field = scan.field(client.ChanField.REFLECTIVITY)
    return range_field







class extraction(object):
    def __init__(self, desired_size = (2048,128), width_crop = [0,2048], 
                pcap_path = "/datasets/lidar_post_detection/raw_lidar_data/channel128_wide_angle/1/lidar/os",
                save_loc = "/datasets/lidar_post_detection/extracted_lidar_data/",
                play_pcap = True,
                extracted = True):
        #this is a nice size for convolution filters as its not TOO big as well as have size in power of 2
        self.desired_size = desired_size
        self.width_crop = width_crop


        self.pcap_path = pcap_path+'.pcap'
        self.metadata_path = pcap_path+'.json'


        normalize_value = [30000,100,100,50] #for 32 channel using 15, for 128 channel using 50
        lidar_channel = 'signal' #use near_ir for detecting the circle
        self.count_frames = True
        self.pause_each = False

        self.save_loc = save_loc
        self.save_every = 10

        self.start_save = 200
        self.end_save = 3000
        self.play_pcap = play_pcap
        self.extracted = extracted

        

        if not self.extracted:
            self.extract_save()


    def extract_image(self, source, scan):
        range_field = load_channel('range', scan)
        near_field = load_channel('near_ir', scan)
        signal_field = load_channel('signal', scan)
        reflect_field = load_channel('reflect', scan)

        #destagger each channel
        range_img = client.destagger(source.metadata, range_field).astype('float')
        near_img = client.destagger(source.metadata, near_field).astype('float')
        signal_img = client.destagger(source.metadata, signal_field).astype('float')
        reflect_img = client.destagger(source.metadata, reflect_field).astype('float')

        #normalize each individual channel
        range_img = ouster_normalize(range_img,0)
        near_img = ouster_normalize(near_img,1)
        #near_img = inverse_norm(near_img)
        #near_img = ring_norm(near_img)
        signal_img = ouster_normalize(signal_img,2)
        reflect_img = ouster_normalize(reflect_img,3)


        multi_img = np.dstack([range_img,signal_img,reflect_img])
        multi_img = cv2.resize(multi_img, self.desired_size)
        multi_img = multi_img[:,self.width_crop[0]:self.width_crop[1]]

        depth_img = range_img

        return multi_img, depth_img

    def extract_save(self):
        if os.path.exists(self.save_loc):
            inp = input("save path location already exists, delete and create fresh ? (y/n) ")
            if inp=='y':
                print("deleting save loc and creating fresh ")
                shutil.rmtree(self.save_loc)
                print("deleted")
                os.makedirs(self.save_loc)
        with open(self.metadata_path, 'r') as f:
            metadata = client.SensorInfo(f.read())

        source = pcap.Pcap(self.pcap_path, metadata)

        scans = iter(client.Scans(source))
        count = 0
        for idx, scan in enumerate(scans):
            count+=1
            if self.count_frames:
                print("Frame number ",count)



            if count>self.start_save: #and count<end_save:
                input_img, depth = self.extract_image(source, scan)
                if count%self.save_every==0:
                    cv2.imwrite(self.save_loc+str(count)+'.png',depth*255.0)

                if self.play_pcap:
                    cv2.imshow(' normalized image ',depth)
                    if self.pause_each:
                        cv2.waitKey(0)
                    else:
                        cv2.waitKey(1)
            
            elif count>self.end_save:
                print("mentioned not to save after this Frame ")
                break

        print("Done extraction !")

    def show_sample_3d(self, imfile, show_bounding_box = True):
        #sdepth = cv2.imread(self.save_loc+"1_450.png",0)
        sdepth = cv2.imread(imfile,0)

        srgb = 0.5*np.ones((sdepth.shape[0],sdepth.shape[1],3))

        #approximate lidar as a camera
        camera_params = {
                            "fx": 800,
                            "fy": 100,
                            "centerX": sdepth.shape[1]//2,
                            "centerY": sdepth.shape[0]//2,
                            "scalingFactor": 1
                        }
        print("time ",dt.now())
        #project_rgbd.show_pcd_from_rgbd(srgb, sdepth, camera_params, save_loc = "")
        #pcd,_ = project_rgbd.project_rgbd_to_pointcloud(srgb, sdepth, camera_params)
        pcd = project_rgbd.pcd_from_rgbd_native(srgb, sdepth,camera_params)
        print("time ",dt.now())

        if show_bounding_box:
            print("showing pcd with bounding boxes ")
            #bb = o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(pcd)
            bb = o3d.geometry.OrientedBoundingBox.get_axis_aligned_bounding_box(pcd)
            #print("got bounding box points ",np.array(bb.get_box_points()))
            bb = o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(bb)
            bb.color = [1,0,0]
            o3d.visualization.draw_geometries([pcd, bb])

            print("got bounding box points ",np.array(bb.get_box_points()))

        project_rgbd.reproject_depth_map(pcd)







if __name__ == '__main__':


    e = extraction(extracted = True)
    
    #e.show_sample_3d()

    sample_imfile = e.save_loc+"480.png"

    ba = cv_annotator.bounding_box_annotator(sample_imfile, desired_size = (-1,-1), normalize_bbox = False)
    bboxes = ba.run()

    img = cv2.imread(sample_imfile,0)
    print("max of loaded image ",np.max(img))
    b = bboxes[0]
    crop = img[b[0]:b[2], b[1]:b[3]]

    
    #idea 
    #use deep lab v3 based finetuning to learn to segment pillars
    #inference->segmentation-> 
    #   mean pixel value of the segmented area -> thresh
    #   contour approximated bounding box -> bounding box label for training efficient object detector
    #   contour approximated bounding box -> distort the bounding box a little bit by changing dimensions/ center coordinates -> generate pairs for training siamese depth estimator
    
    #   siamese depth estimator takes as input two 2D images denoting possibly containing the pillar within it, but the box may be innacurate and not tight
    #   Siamese depth estimator output -> tight box containing pillar within + 3 additional values -> thresh, z_plus, z_minus

    #whole pipeline -> input projected lidar image -> object detection -> interleaved tracking + siamese depth regressor -> real time 3D bounding box output

    '''
    x_train = crop.reshape(-1,1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x_train)
    l = kmeans.labels_
    contained_dists = x_train[l == 0]
    thresh = np.mean(contained_dists)  #max of the loaded image is 255.0/ sample good - 64
    z_plus, z_minus = 10, 10

    print("found threshold ",thresh)
    print("using zplus zminus ",z_plus,z_minus)
    crop[crop<thresh-z_minus] = 0.0
    crop[crop>thresh+z_plus] = 0.0
    '''
    
    
    

    cv2.imshow("cropped image ",crop)
    cv2.waitKey(0)
    cv2.imwrite("temp.png",crop)

    e.show_sample_3d("temp.png")

    


