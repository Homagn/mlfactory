import tensorflow as tf
import numpy as np
import pandas as pd
import json
from tqdm.notebook import tqdm; tqdm.pandas()
import cv2

import os,sys

############################################
#script to download the coco dataset
#(from)
#https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9


'''
mkdir coco
cd coco
mkdir images
cd images

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip
wget -c http://images.cocodataset.org/zips/unlabeled2017.zip

unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip unlabeled2017.zip

rm train2017.zip
rm val2017.zip
rm test2017.zip
rm unlabeled2017.zip 

cd ../
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip
unzip image_info_test2017.zip
unzip image_info_unlabeled2017.zip

rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip
rm image_info_test2017.zip
rm image_info_unlabeled2017.zip
'''

############################################
############################################







data_path = '/datasets/coco/'
#data_location_val = data_path + 'annotations/instances_val2017.json'
data_location_train = data_path + 'annotations/instances_train2017.json'



class_map = {"person":0}
object_class_to_train = "person"
mask_cat_id = class_map[object_class_to_train]



path = data_path
train_path = path + 'images/train2017/'
val_path = path + 'images/val2017/'
test_path = path + 'images/test2017/'



#Object detection data format 
# base_folder/annotations/instances_train.json - contains annotations as a dictionary. 
# To understand, see this - https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
# base_folder/images/train_2017/ -  contains all the raw images 





################################ Get MS COCO dataset ready ########################
###################################################################################
###################################################################################
###################################################################################


def coco_to_yolo(bbox, img_w, img_h):
    bbox = np.array(bbox)
    bbox[:,0:1] = (bbox[:, 0:1] + bbox[:, 2:3]/2.)/img_w
    bbox[:, 1:2] = (bbox[:, 1:2] + bbox[:, 3:4]/2.)/img_h
    bbox[:, 2:3] = bbox[:, 2:3]/img_w
    bbox[:, 3:4] = bbox[:, 3:4]/img_h
    return bbox.tolist()

def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2




class coco_dataframe_loader(object):
    def __init__(self):

        if os.path.exists("/datasets/coco/dataframe.pkl"):
            inp = input("Dataframe file already exists, load it ? (y/n) ")
            if inp=="y":
                self.train_df = pd.read_pickle("/datasets/coco/dataframe.pkl")
                print("Dataframe file loaded from previously made coco dataframe ")
                return

        with open(data_location_train) as f:
            annot_train = json.load(f)

        '''
        with open(data_location_val) as f:
            annot_val = json.load(f)
        '''

        mapper=dict([list(d.values())[1:] for d in annot_train['categories']])
        id_mapper = dict([(a, b) for a, b in zip(mapper.keys(), np.arange(len(mapper)))])
        mapper = dict([(a, b) for a, b in zip(np.arange(len(mapper)), mapper.values())])

        train_annot_df = pd.DataFrame(annot_train['annotations'])
        train_annot_df['category_id'] = train_annot_df.category_id.apply(lambda x: id_mapper[x])
        train_annot_df['category_id'] = train_annot_df.category_id.astype('int32')
        train_annot_df = train_annot_df.groupby('image_id')['category_id','bbox'].agg(list).reset_index()
        train_image_df = pd.DataFrame(annot_train['images'])
        train_image_df.rename(columns={'id':'image_id'}, inplace=True)
        train_df = pd.merge(train_annot_df, train_image_df, how='right', right_on='image_id', left_on='image_id')
        train_df['file_name'] = train_df.file_name.progress_apply(lambda x: train_path+x)

        train_df.fillna('nan', inplace=True)
        train_df['bbox'] = train_df.bbox.apply(lambda x: x if x!='nan' else [[0,0,0,0]])
        train_df['yolo_bbox'] = train_df[['bbox', 'width', 'height']].apply(lambda x: coco_to_yolo(x.bbox, x.width, x.height), axis=1)
        train_df['image_id'] = train_df.image_id.astype('int32')
        train_df['height'] = train_df.height.astype('float32')
        train_df['width'] = train_df.width.astype('float32')
        train_df.drop(['license', 'coco_url', 'date_captured', 'flickr_url','bbox'], axis=1, inplace=True)
        train_df['category_id'] = train_df.category_id.apply(lambda x: x if x!='nan' else [0])
        print('TRAINING DATAFRAME CREATION COMPLETED')

        '''
        val_annot_df = pd.DataFrame(annot_val['annotations'])
        val_annot_df['category_id'] = val_annot_df.category_id.apply(lambda x: id_mapper[x])
        val_annot_df['category_id'] = val_annot_df.category_id.astype('int32')
        val_annot_df = val_annot_df.groupby('image_id')['category_id','bbox'].agg(list).reset_index()
        val_image_df = pd.DataFrame(annot_val['images'])
        val_image_df.rename(columns={'id':'image_id'}, inplace=True)
        val_df = pd.merge(val_annot_df, val_image_df, how='right', right_on='image_id', left_on='image_id')
        val_df['file_name'] = val_df.file_name.progress_apply(lambda x: val_path+x)

        val_df.fillna('nan', inplace=True)
        val_df['bbox'] = val_df.bbox.apply(lambda x: x if x!='nan' else [[0,0,0,0]])
        val_df['yolo_bbox'] = val_df[['bbox', 'width', 'height']].apply(lambda x: coco_to_yolo(x.bbox, x.width, x.height), axis=1)
        val_df['image_id'] = val_df.image_id.astype('int32')
        val_df['height'] = val_df.height.astype('float32')
        val_df['width'] = val_df.width.astype('float32')
        val_df.drop(['license', 'coco_url', 'date_captured', 'flickr_url'], axis=1, inplace=True)
        val_df['category_id'] = val_df.category_id.apply(lambda x: x if x!='nan' else [0])

        print('VALIDATION DATAFRAME CREATION COMPLETED')
        '''

        print("checking dataframes ")
        print("train_df table ",train_df['file_name'])

        #make sure we only sample images that contain the category we want
        category_mask = train_df.category_id.apply(lambda x: mask_cat_id in x)
        self.train_df = train_df[category_mask]

        print("saving the dataframe ")
        self.train_df.to_pickle("/datasets/coco/dataframe.pkl") 

        

    def sample(self):
        batch = self.train_df.sample(n=4, replace=True, random_state=2)

        print("checking dataframes ")
        '''
        print("train_df file names ",train_df.at[100,'file_name'])
        print("train_df boxes ",train_df.at[100,'yolo_bbox'])
        print("train_df categories ",train_df.at[100,'category_id'])
        '''
        fname_batch = batch['file_name'].values.tolist()
        bbox_batch = batch['yolo_bbox'].values.tolist()
        cat_batch = batch['category_id'].values.tolist()

        #print("train_df file names ",fname_batch)
        #print("train_df boxes ",bbox_batch)
        #print("train_df categories ",cat_batch)

        return fname_batch, bbox_batch, cat_batch


    def sample_targets(self, batch_size = 4 ):
        batch = self.train_df.sample(n=batch_size, replace=True, random_state=1)
        fname_batch = batch['file_name'].values.tolist()
        bbox_batch = batch['yolo_bbox'].values.tolist()
        cat_batch = batch['category_id'].values.tolist()

        color = (255, 0, 255)
        thickness = 1

        bbox = bbox_batch
        file_name = fname_batch

        x = []
        y = []

        imgs = []


        for b in range(batch_size):
            img = cv2.imread(file_name[b])
            dw = img.shape[1]
            dh = img.shape[0]
            x.append(img)
            yi = []

            bbox = bbox_batch[b]
            cats = cat_batch[b]

            for idx in range(len(bbox)):

                i = bbox[idx]
                if cats[idx]!=mask_cat_id: #a single image can contain both desired and undesired category labels, just chose the one we want
                    continue
                top_left = (int( (i[0] - (i[2]/2) )*dw), int( (i[1] - (i[3]/2) )*dh) )
                bottom_right = ( int( (i[0]+(i[2]/2) )*dw )  , int( (i[1]+(i[3]/2))*dh  ) )

                yi.append( np.array([ (i[0] - (i[2]/2) ), (i[1] - (i[3]/2) ) , (i[0]+(i[2]/2) ) ,  (i[1]+(i[3]/2))  ], dtype = np.float32) )

                img = cv2.rectangle(img, top_left, bottom_right, color, thickness)

            imgs.append(img)
            y.append(np.array(yi))

        return np.array(x), np.array(y)



    def viz_sample(self, file_name, bbox, cats):
        #NOTE - i[0], i[1] are the center x and center y of the bounding box scaled with respect to the width and height of the image
        #     - i[2], i[3] are the width and height of the box scaled with respect to the width and height of the image

        img = cv2.imread(file_name)
        color = (255, 0, 255)
        thickness = 1
        img_w = img.shape[1]
        img_h = img.shape[0]

        dw = img.shape[1]
        dh = img.shape[0]
        print("got image width and height ",img_w, img_h)
        print("got targets ",bbox)
        
        for idx in range(len(bbox)):

            i = bbox[idx]
            if cats[idx]!=mask_cat_id: #a single image can contain both desired and undesired category labels, just chose the one we want
                continue
            top_left = (int( (i[0] - (i[2]/2) )*dw), int( (i[1] - (i[3]/2) )*dh) )
            bottom_right = ( int( (i[0]+(i[2]/2) )*dw )  , int( (i[1]+(i[3]/2))*dh  ) )

            img = cv2.rectangle(img, top_left, bottom_right, color, thickness)

        cv2.imshow("image with box ",img)
        cv2.waitKey(0)


class centernet_sampler(object):
    def __init__(self):
        self.cdl = coco_dataframe_loader()
    def sample(self):
        file_names, boxes, categories = self.cdl.sample()
        xb, yb = self.cdl.sample_targets()
        
        print("got boxes ",yb)
        sys.exit(0)

        boxes = np.array(boxes)
        center = np.asarray([np.mean(np.asarray([boxes[:,2],boxes[:,0]]), axis=0).T, np.mean(np.asarray([boxes[:,3],boxes[:,1]]), axis=0).T]).T
        print("got targets ",boxes)
        print("got centers ",center)
        







################################ Visualization check ##############################
###################################################################################
###################################################################################
###################################################################################

'''
cdl = coco_dataframe_loader()
file_names, boxes, categories = cdl.sample()

idx = 0
cdl.viz_sample(file_names[idx],boxes[idx], categories[idx])
'''


cns = centernet_sampler()
cns.sample()


