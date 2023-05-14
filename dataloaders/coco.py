import tensorflow as tf
import numpy as np
import pandas as pd
import json
from tqdm.notebook import tqdm; tqdm.pandas()
import cv2



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


data_location_val = data_path + 'annotations/instances_val2017.json'
data_location_train = data_path + 'annotations/instances_train2017.json'



class_map = {"person":0}
object_class_to_train = "person"
mask_cat_id = class_map[object_class_to_train]



#to do - put it into standard dataloader class structure





################################ Get MS COCO dataset ready ########################
###################################################################################
###################################################################################
###################################################################################

#path = f'../input/coco-2017-dataset/coco2017/'
path = data_path
train_path = path + 'images/train2017/'
val_path = path + 'images/val2017/'
test_path = path + 'images/test2017/'

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

with open(data_location_train) as f:
    annot_train = json.load(f)
with open(data_location_val) as f:
    annot_val = json.load(f)
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


print("checking dataframes ")
print("train_df table ",train_df['file_name'])

category_mask = train_df.category_id.apply(lambda x: mask_cat_id in x)
train_df = train_df[category_mask]

batch = train_df.sample(n=4, replace=True, random_state=1)
#category_mask = batch.category_id.apply(lambda x: mask_cat_id in x)
#batch = batch[category_mask]

print("checking dataframes ")
'''
print("train_df file names ",train_df.at[100,'file_name'])
print("train_df boxes ",train_df.at[100,'yolo_bbox'])
print("train_df categories ",train_df.at[100,'category_id'])
'''
fname_batch = batch['file_name'].values.tolist()
bbox_batch = batch['yolo_bbox'].values.tolist()
cat_batch = batch['category_id'].values.tolist()

print("train_df file names ",fname_batch)
print("train_df boxes ",bbox_batch)
print("train_df categories ",cat_batch)

def plot_img_and_bbox(file_name, bbox, cats):
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

    
    for idx in range(len(bbox)):

        i = bbox[idx]
        if cats[idx]!=0:
            continue
        top_left = (int( (i[0] - (i[2]/2) )*dw), int( (i[1] - (i[3]/2) )*dh) )
        bottom_right = ( int( (i[0]+(i[2]/2) )*dw )  , int( (i[1]+(i[3]/2))*dh  ) )

        img = cv2.rectangle(img, top_left, bottom_right, color, thickness)

    cv2.imshow("image with box ",img)
    cv2.waitKey(0)


################################ Visualization check ##############################
###################################################################################
###################################################################################
###################################################################################
plot_img_and_bbox(fname_batch[2],bbox_batch[2], cat_batch[2])
print("exiting ...")
sys.exit(0)


################################ Data prep ########################################
###################################################################################
###################################################################################
###################################################################################

IMG_SIZE = 640
BATCH = 8
AUTO = tf.data.AUTOTUNE

def prepare_data(file_path, bbox, category, w, h, img_size):
    image = tf.image.decode_jpeg(tf.io.read_file(file_path), channels=3)
    image = tf.cast(image, tf.float32)/255.0
    w = tf.cast(w, tf.float32)
    h = tf.cast(h, tf.float32)
    bbox = tf.reshape(bbox.to_tensor(), [-1,4])
    image = tf.image.resize(image, img_size)
    category = tf.cast(category, tf.float32)
    label  = tf.concat([bbox, category[...,tf.newaxis]],-1)
    #label = anchor_labeler.encode(label)
    return image, label

def build_dataset(df, img_size=(IMG_SIZE, IMG_SIZE), train=True):
    dataset = tf.data.Dataset.from_tensor_slices((df.file_name, tf.ragged.constant(df.yolo_bbox),
                                                   tf.ragged.constant(df.category_id), df.width, df.height))
    dataset = dataset.map((lambda f_n, box, cl, w, h: prepare_data(f_n, box, cl, w, h, img_size=img_size)), num_parallel_calls=AUTO)
    dataset = dataset.padded_batch(BATCH, drop_remainder=True)
    return dataset


def sample(dataset):
    d = dataset.take(1)
    #print("dataset ",list(d.as_numpy_iterator()))
    #print("dataset ",np.array( list( d.as_numpy_iterator() ) [0] ).shape)
    print("dataset ", list( d.as_numpy_iterator() ) [0][0].shape) #(8,640,640,3)
    print("dataset ", list( d.as_numpy_iterator() ) [0][1].shape) #(8,24,5)
    return



train_ds = build_dataset(train_df)
val_ds = build_dataset(val_df)

print("Done data prep ")

sample(train_ds)