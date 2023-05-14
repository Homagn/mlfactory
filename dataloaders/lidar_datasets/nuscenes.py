#dataset - go to nuscenes website and downloads

#for labels follow this code
#get the indices from - /home/homagni/Downloads/nuScenes-lidarseg-all-v1.0/v1.0-mini/category.json

import numpy as np
import struct
from open3d import *
import open3d as o3d
import glob
import json
from os.path import exists
import cv2
from more_itertools import locate # pip install more-itertools
import math

import time

import os
import sys
import copy

os.environ['top'] = '../../'
sys.path.append(os.path.join(os.environ['top']))

from misctools.lidar_camera_align import align_funcs
from misctools.rgbd_mapping import project_rgbd

manual_align_pcd = align_funcs.manual_align_pcd
sensor_alignment = align_funcs.sensor_alignment
crop_fov = align_funcs.crop_fov


project_rgbd_to_pointcloud = project_rgbd.project_rgbd_to_pointcloud


dataset_name = '/datasets/nuscenes/'
data_part = 1




def find_indices(list_to_check, item_to_find):
    indices = locate(list_to_check, lambda x: x == item_to_find)
    return list(indices)


def gen_colorings(num_unique):
    colors = np.abs(np.random.random_sample((num_unique,3)))
    clist = [i.tolist() for i in colors]
    return clist


def box_rotation(rmat):
    #bb is o3d geometry type oriented bounding box
    
    finalRot = copy.copy(np.linalg.inv(rmat))
    det = np.linalg.det(rmat) 
    if det < 0: 
        finalRot = np.multiply(finalRot, reflect) 
    return finalRot

def show_pcd(pcds):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for pcd in pcds:
        vis.add_geometry(pcd)
    # run visualizer main loop
    print("Press Q or Excape to exit")
    vis.run()
    vis.destroy_window()





def create_pcd_from_points(points,colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd



def get_pcd_parts(pcd_name, seg_name, class_ids): #if class_ids = [], then whole pointcloud has the same color, also pcd_parts has only one key 0 storing the entire pointcloud
    if '.bin' in seg_name: #loading the lidarseg
        seg=np.fromfile(seg_name, dtype=np.uint8)
    if '.npz' in seg_name: #panoptic
        seg = np.load(seg_name)['data']


    

    #print("uniques labels in the pointcloud ",np.unique(seg), " total number ",len(np.unique(seg)) )

    #assign a brownish color
    background_color = [0.4,0.3,0.3]
    colors = np.array(background_color*len(seg)).reshape((len(seg),3))
    #the entire pointcloud
    scan=np.fromfile(pcd_name, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :4]
    points = points[:, :3]

    
    pcd_parts = {}


    us = np.unique(seg)

    for class_id in class_ids:
        if '.npz' in seg_name: #panoptic
            usc = us[us//1000==class_id]
        if '.bin' in seg_name:
            usc = us[us==class_id]
        #print("got unique labels for the class ",usc)

        if class_id not in pcd_parts.keys():
            pcd_parts[class_id] = []


        instance_colorings = gen_colorings( len(usc) )
        for u in range(len(usc)): #if semantic labels than len(usc) is always 1, if panotic then can be more than 1 depending on number of instances
            p = points[seg==usc[u]]
            num_p = p.shape[0]
            c = np.array(instance_colorings[u]*num_p).reshape((num_p,3))

            inst_pcd = create_pcd_from_points(p,c)

            pcd_parts[class_id].append(inst_pcd)

            colors[seg==usc[u],:] = instance_colorings[u]


    #key -1 in pcd_parts store all the points which does not belong to the interest classes
    p_rest = points[(colors[:,0]==background_color[0])&(colors[:,1]==background_color[1])&(colors[:,2]==background_color[2])]
    c_rest = colors[(colors[:,0]==background_color[0])&(colors[:,1]==background_color[1])&(colors[:,2]==background_color[2])]
    inst_pcd = create_pcd_from_points(p_rest,c_rest)
    pcd_parts[-1] = [inst_pcd]


    pcd_full = create_pcd_from_points(points,colors)
    pcd_parts[0] = pcd_full

    return pcd_parts

            
def join_pcd_parts(pcd_parts, class_id):
    all_points = []
    all_colors = []

    for k in pcd_parts.keys():
        if k==class_id:
            for sub_pcd in pcd_parts[k]:
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





        

def get_annotation_boxes():
    #How to
    #https://github.com/nutonomy/nuscenes-devkit/blob/84ee1f3ad767a0ac5953fdd2f2d622bef46bedb0/python-sdk/nuscenes/nuscenes.py#L834
    #see from line 319
    pass



def load_meta_files(label_files_json,meta_files_json, calibrated_sensor_json, ego_pose_json):
    print("opening big json files please wait ...")
    # Opening JSON file
    f = open(label_files_json)
    label_data = json.load(f)
    f.close()

    g = open(meta_files_json)
    meta_data = json.load(g)
    g.close()

    h = open(calibrated_sensor_json)
    calib_sensor_data = json.load(h)
    h.close()

    j = open(ego_pose_json)
    ego_pose_data = json.load(j)
    j.close()

    return label_data, meta_data, calib_sensor_data, ego_pose_data



def get_lidar_files_nth(label_data, meta_data, calib_sensor_data, ego_pose_data, labels_folder, data_part, idx):
    #print("done reading json files, now finding proper aligning rgb file ")

    meta_tokens = [ m['token'] for m in meta_data ]
    calib_tokens = [ c['token'] for c in calib_sensor_data ]
    ego_tokens = [ e['token'] for e in ego_pose_data ]
    meta_sample_tokens = [m['sample_token'] for m in meta_data]


    pcd_location, pcd_seg_location = None, None

    #print("number of lidar files in download data part ",len(label_data))
    if idx>len(label_data):
        print("data not downloaded index is higher than range ")
        return None,None

    count = 0
    index = 0
    while count!=idx+1:
        
        while (True):
            #print("trying")
            token_id = label_data[index]['token']
            pcd_seg_location = labels_folder+label_data[index]['filename']


            
            mtidx = meta_tokens.index(token_id) 

            pcd_filename = meta_data[ mtidx ]["filename"]
            pcd_timestamp = meta_data[ mtidx ]["timestamp"] #used to find closest other sensor capture time frame
            token_sample_id = meta_data[ mtidx ]['sample_token'] #used for finding corresponding camera frame tokens

            pcd_location = dataset_name+"part"+str(data_part)+"/"+"v1.0-trainval01_blobs_lidar/"+pcd_filename


            index+=1

            if exists(pcd_location):
                #print("pcd location exists ",pcd_location)
                #print("found matching downloaded pcd file at index ",index)
                count+=1
                break
            
    

    '''
    print("\n\n")
    print("file location information \n")
    print("pcd_location ",pcd_location)
    print("\n")
    print("pcd_seg_location ",pcd_seg_location)
    print("\n")
    '''
    return pcd_location, pcd_seg_location













def get_all_modality_sample(label_files_json,meta_files_json, calibrated_sensor_json, ego_pose_json, labels_folder, data_part):
    print("opening big json files please wait ...")
    # Opening JSON file
    f = open(label_files_json)
    label_data = json.load(f)
    f.close()

    g = open(meta_files_json)
    meta_data = json.load(g)
    g.close()

    h = open(calibrated_sensor_json)
    calib_sensor_data = json.load(h)
    h.close()

    j = open(ego_pose_json)
    ego_pose_data = json.load(j)
    j.close()





    print("done reading json files, now finding proper aligning rgb file ")

    meta_tokens = [ m['token'] for m in meta_data ]
    calib_tokens = [ c['token'] for c in calib_sensor_data ]
    ego_tokens = [ e['token'] for e in ego_pose_data ]

    meta_sample_tokens = [m['sample_token'] for m in meta_data]

    tries = 0
    while True: #only part 1 of the data has been downloaded so trying multiple times until the matching token and downloaded data is found

        idx = np.random.choice(len(label_data),1)[0]
        token_id = label_data[idx]['token']
        
        pcd_seg_location = labels_folder+label_data[idx]['filename']


        
        mtidx = meta_tokens.index(token_id) 

        pcd_filename = meta_data[ mtidx ]["filename"]
        pcd_timestamp = meta_data[ mtidx ]["timestamp"] #used to find closest other sensor capture time frame
        token_sample_id = meta_data[ mtidx ]['sample_token'] #used for finding corresponding camera frame tokens

        

        
        


        other_sensors_idx = find_indices(meta_sample_tokens, token_sample_id)
        other_sensors_filenames = [meta_data[o]["filename"] for o in other_sensors_idx]
        other_sensors_timestamps = [meta_data[o]["timestamp"] for o in other_sensors_idx]

        other_sensor_tokens = [meta_data[o]["calibrated_sensor_token"] for o in other_sensors_idx]
        other_pose_tokens = [meta_data[o]["ego_pose_token"] for o in other_sensors_idx]

        #print("got number of other sensors filenames ",len(other_sensors_filenames))

        cam_locations = []

        cam_sensor_tokens = []
        cam_pose_tokens = []


        cam_times = []
        for c in range(len(other_sensors_filenames)):
            if '__CAM_FRONT__' in other_sensors_filenames[c]:
                cam_locations.append(dataset_name+"part"+str(data_part)+"/"+"v1.0-trainval01_blobs_camera/"+other_sensors_filenames[ c ])

                cam_sensor_tokens.append(other_sensor_tokens[c])
                cam_pose_tokens.append(other_pose_tokens[c])

                cam_times.append(math.fabs(other_sensors_timestamps[c]-pcd_timestamp))

        #print("minimum lidar camera capture time difference ",min(cam_times))
        min_tdiff_idx = np.argmin(cam_locations)
        cam_locations = [cam_locations[min_tdiff_idx ] ]

        





        #code for projecting lidar to camera
        lidar_sensor_token = meta_data[ mtidx ]["calibrated_sensor_token"] #used for projecting lidar to camera
        lidar_pose_token = meta_data[ mtidx ]["ego_pose_token"] #used for projecting lidar to camera

        cam_sensor_token = cam_sensor_tokens[min_tdiff_idx]
        cam_pose_token = cam_pose_tokens[min_tdiff_idx]










        
        pcd_location = dataset_name+"part"+str(data_part)+"/"+"v1.0-trainval01_blobs_lidar/"+pcd_filename


        if exists(pcd_location):
            #print("pcd location exists ",pcd_location)
            break
        tries+=1
    

    
    cs_record_lidar = calib_sensor_data[ calib_tokens.index(lidar_sensor_token) ]
    #print("retrieved calib sensor data of the lidar as ",cs_record_lidar)
    pose_record_lidar = ego_pose_data[ ego_tokens.index(lidar_pose_token) ]
    #print("retrieved ego pose data of the lidar as ",pose_record_lidar)


    cs_record_cam = calib_sensor_data[ calib_tokens.index(cam_sensor_token) ]
    #print("retrieved calib sensor data of the camera as ",cs_record_cam)
    pose_record_cam = ego_pose_data[ ego_tokens.index(cam_pose_token) ]
    #print("retrieved ego pose data of the camera as ",pose_record_cam)


    print("\n\n")
    print("file location information \n")
    print("pcd_location ",pcd_location)
    print("\n")
    print("pcd_seg_location ",pcd_seg_location)
    print("\n")
    print("cam_locations ",cam_locations)
    print("\n")
    print("cs_record_lidar ",cs_record_lidar)
    print("\n")
    print("pose_record_lidar ",pose_record_lidar)
    print("\n")
    print("cs_record_cam ",cs_record_cam)
    print("\n")
    print("pose_record_cam ",pose_record_cam)
    print("\n")

    return pcd_location, pcd_seg_location, cam_locations, cs_record_lidar, pose_record_lidar, cs_record_cam, pose_record_cam, tries


if __name__ == '__main__':

    #params for the code usage
    #============
    random_sample = 2 #2 sample filelocations are included in later parts of the code if -1, then it randomly samples from downloaded data which takes time
    #below if true then loads an interactive visualization where you can manually align the lidar pointcloud to the rgb image and dont care about class ids
    #if false then dont care about rgbd alignment, load the pointcloud and its instance segmentation and also draw bounding boxes for the specified class ids
    align_to_rgb = False 
    class_ids = [17, 24, 28, 30] #cars-17 (15 to 23 idx are all 4 wheelers) /flat drivable surface - 24 (get these from category.json file)/ static manmade -28/ static vegetation - 30
    #For which classes in the lidar pointcloud you want to draw bounding boxes
    draw_object_box_classes = [17]
    #==============


    #places linking the downloaded dataset to the loading function: (commented options for lidar seg)

    #label_files_json = dataset_name+'nuScenes-lidarseg-all-v1.0/v1.0-trainval/lidarseg.json'
    label_files_json = dataset_name+'nuScenes-panoptic-v1.0-all/v1.0-trainval/panoptic.json'
    meta_files_json = dataset_name+'v1.0-trainval_meta/v1.0-trainval/sample_data.json'
    calibrated_sensor_json = dataset_name+'v1.0-trainval_meta/v1.0-trainval/calibrated_sensor.json'
    ego_pose_json = dataset_name+'v1.0-trainval_meta/v1.0-trainval/ego_pose.json'
    #labels_folder = dataset_name+'nuScenes-lidarseg-all-v1.0/'
    labels_folder = dataset_name+'nuScenes-panoptic-v1.0-all/'

    

    
    if align_to_rgb:
        print("should not draw bounding boxes when aligning to RGB because pcd has been cropped")
        draw_object_box_classes = []







    if random_sample==-1:
        pcd_location, pcd_seg_location, cam_locations, cs_record_lidar, pose_record_lidar, cs_record_cam, pose_record_cam, tries = get_all_modality_sample(label_files_json,meta_files_json, calibrated_sensor_json, ego_pose_json, labels_folder)
    else:
        #load a known file quickly
        tries = 0

        #Example 1
        if random_sample==1:
            pcd_location = '/datasets/nuscenes/part1/v1.0-trainval01_blobs_lidar/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151482696991.pcd.bin'
            pcd_seg_location = '/datasets/nuscenes/nuScenes-panoptic-v1.0-all/panoptic/v1.0-trainval/8c5118a1d7d44b4991703190d47dff5e_panoptic.npz'
            cam_locations = ['/datasets/nuscenes/part1/v1.0-trainval01_blobs_camera/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151482662404.jpg']

            cs_record_lidar = {'token': 'b0b8e83f3b8d44569285cddc1dc1402b', 'sensor_token': 'dc8b396651c05aedbb9cdaae573bb567', 'translation': [0.985793, 0.0, 1.84019], 'rotation': [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719], 'camera_intrinsic': []}
            pose_record_lidar = {'token': '8c5118a1d7d44b4991703190d47dff5e', 'timestamp': 1533151482696991, 'rotation': [-0.26579334928644377, 6.072434003397313e-05, -0.0034480623648620004, -0.9640238600022184], 'translation': [593.42761833282, 1511.444527035144, 0.0]}

            cs_record_cam = {'token': '6aa8968106b4486bb4740b591e6f626c', 'sensor_token': '725903f5b62f56118f4094b46a4470d8', 'translation': [1.72200568478, 0.00475453292289, 1.49491291905], 'rotation': [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754], 'camera_intrinsic': [[1252.8131021185304, 0.0, 826.588114781398], [0.0, 1252.8131021185304, 469.9846626224581], [0.0, 0.0, 1.0]]}
            pose_record_cam = {'token': '201b16c5793441eeb67d145ce3a0aeac', 'timestamp': 1533151482662404, 'rotation': [-0.26578656024209, 6.257040996366091e-05, -0.003436476146609336, -0.9640257730534574], 'translation': [593.4276182746739, 1511.4445269946643, 0.0]}
        

        #Example 2
        if random_sample==2:
            pcd_location = '/datasets/nuscenes/part1/v1.0-trainval01_blobs_lidar/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151426148005.pcd.bin'
            pcd_seg_location = '/datasets/nuscenes/nuScenes-panoptic-v1.0-all/panoptic/v1.0-trainval/bc9afe526799415dad5a9a2ef4fa7304_panoptic.npz'
            cam_locations = ['/datasets/nuscenes/part1/v1.0-trainval01_blobs_camera/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151426112404.jpg']

            cs_record_lidar = {'token': '1f28713845f6416cab99138c275093a9', 'sensor_token': 'dc8b396651c05aedbb9cdaae573bb567', 'translation': [0.985793, 0.0, 1.84019], 'rotation': [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719], 'camera_intrinsic': []}
            pose_record_lidar = {'token': 'bc9afe526799415dad5a9a2ef4fa7304', 'timestamp': 1533151426148005, 'rotation': [0.18981546627945772, 0.0003377098599420826, 0.0014378155144021943, 0.9818186733810426], 'translation': [725.6791611974511, 1431.9342779221583, 0.0]}

            cs_record_cam = {'token': '3e512dd067954f50ad0f4f796b6dd8db', 'sensor_token': '725903f5b62f56118f4094b46a4470d8', 'translation': [1.72200568478, 0.00475453292289, 1.49491291905], 'rotation': [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754], 'camera_intrinsic': [[1252.8131021185304, 0.0, 826.588114781398], [0.0, 1252.8131021185304, 469.9846626224581], [0.0, 0.0, 1.0]]}
            pose_record_cam = {'token': 'f84ced75f1b9410282231b3e6a0bc2fe', 'timestamp': 1533151426112404, 'rotation': [0.18636664326258273, 0.000296864560924093, 0.0015020831052054194, 0.9824790735159782], 'translation': [725.8598335710717, 1431.8614417519389, 0.0]}





    bin_file = pcd_location
    label_file = pcd_seg_location


    print("number of tries ",tries)
    print("pcd raw file location ",pcd_location)
    print("pcd annotated location ",pcd_seg_location)

    print("number of camera captures ",len(cam_locations))
    for c in cam_locations:
        print("got camera location ",c)
        img = cv2.imread(c)
        cv2.imshow("camfront ",img)
        cv2.waitKey(0)
    


    pcd_parts = get_pcd_parts(bin_file, label_file, class_ids)
    
    pcd_ = pcd_parts[0] #0 is always the full pcd, other indices stores class index specific pcds only
    
    #can get the pcd containing points from only a specific type of object in the scene example id 17 (cars)
    #pcd_ = join_pcd_parts(pcd_parts,17)



    if align_to_rgb:
        pcd_ = sensor_alignment(pcd_, cs_record_lidar,pose_record_lidar, cs_record_cam, pose_record_cam)
        pcd_ = crop_fov(pcd_, depth_cutoff = -6)
        #loads up an interactive pointcloud visualizer which can be controlled using keyboard to manually align the lidar points to camera image using visual comparisons
        manual_align_pcd(pcd_)

        camera_params = {}
        camera_params["fx"] = cs_record_cam['camera_intrinsic'][0][0]
        camera_params["fy"] = cs_record_cam['camera_intrinsic'][1][1]
        camera_params["centerX"] = cs_record_cam['camera_intrinsic'][0][2]
        camera_params["centerY"] = cs_record_cam['camera_intrinsic'][1][2]
        camera_params["scalingFactor"] = 1.0

        lidar2cam_pcd, _ = project_rgbd_to_pointcloud(cam_locations[0],"capture_depth_buffer.png", camera_params )


        show_pcd([lidar2cam_pcd])

    else:
        show_pcd([pcd_])



    if draw_object_box_classes!=[]:
        
        to_show = [pcd_]

        class_id = draw_object_box_classes[0]


        bb = o3d.geometry.OrientedBoundingBox.get_axis_aligned_bounding_box(pcd_parts[0])
        box_corners = np.asarray(bb.get_box_points())
        print("got bounding box for the whole pcd ",box_corners)

        for pcd in pcd_parts[class_id]:
            try:
                #oriented box sometimes fits a bad orientation bounding box 
                #aligned fits a box always aligned to x,y,z axis its not tight but it looks good

                #bb = o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(pcd)
                bb = o3d.geometry.OrientedBoundingBox.get_axis_aligned_bounding_box(pcd)
                to_show.append(bb)
                box_corners = np.asarray(bb.get_box_points())
                #print("got box corner points ",box_corners)

            except:
                print("not enough points were there to construct the convex hull")

        show_pcd(to_show)


    '''
    Note (for align_to_rgb=True)
    Run the code
    after the camera image shows up press esc
    now the lidar pcd shows up, press L to load the aligment
    now press C to capture the depth buffer
    now press esc
    after some time the RGBD point cloud will load up
    '''



    




