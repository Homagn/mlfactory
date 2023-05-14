from glob import glob
import shutil
import argparse
import zipfile
import hashlib
import requests
from tqdm import tqdm
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import cv2
import copy

import torch
import torch.nn.functional as F

os.environ['top'] = '../'
sys.path.append(os.path.join(os.environ['top']))

from dataloaders.utils import augmentations
import sys




path = "/datasets"
download_folder = "ADE20K"
scene_names_file = "sceneparsing"

# some helper functions to download the dataset
# this code comes mainly from gluoncv.utils
def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]

def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname

def download_ade(path, overwrite=False):

    """Download ADE20K
    Parameters
    ----------
    path : str
      Location of the downloaded files.
    overwrite : bool, optional
      Whether to overwrite destination file if already exists.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    _AUG_DOWNLOAD_URLS = [
      ('http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip', '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),
      ('http://data.csail.mit.edu/places/ADEchallenge/release_test.zip', 'e05747892219d10e9243933371a497e905a4860c'),]
    
    download_dir = os.path.join(path, download_folder)
    
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall(path=download_dir)


#download_ade(path, overwrite=False)

def read_files():
    path = "/datasets/ADE20K/ADEChallengeData2016/"
    train_images = sorted(glob(os.path.join(path, "images/training/*")))
    train_masks = sorted(glob(os.path.join(path, "annotations/training/*")))

    data = {"images":train_images, "masks":train_masks}

    return data, len(data["images"])

def load(image_path, mask_path):
    x = cv2.imread(image_path)
    y = cv2.imread(mask_path,0) #force read grayscale

    return x, y


def interactive_viz(rgb,mask,objectnames_folder):
    object_names = np.genfromtxt(objectnames_folder+"/objectInfo150.csv", delimiter=',', dtype = 'str')
    names = {i[0]:i[-1] for i in object_names}
    #print("names ",names)

    cv2.imshow("rgb", rgb)
    cv2.waitKey(0)

    for u in np.unique(mask):
        if u ==0: #id 0 is unannotated
            continue
        um = np.zeros_like(mask)
        print("object id ",u," class name ",names[str(u)])
        select = np.where(mask!=u,um,255.0)

        #heatmap = cv2.applyColorMap(um, cv2.COLORMAP_HSV)
        #cv2.imshow("mask",heatmap)
        cv2.imshow("mask ",select)
        cv2.waitKey(0)

class dataloader(object):
    def __init__(self, n_classes=1):
        self.batch_size = 8
        self.w = 256
        self.h = 256

        self.w_target = 256
        self.h_target = 256

        self.n_classes = n_classes
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        download_dir = os.path.join(path, download_folder)
        if not os.path.exists(download_dir):
            print("ade20k dataset is not downloaded, downloading now ...")
            download_ade(path, overwrite=False)
        else:
            print("Dataset has been already downloaded ")

        self.objectnames = os.path.join(download_dir, scene_names_file)
        if not os.path.exists(self.objectnames):
            print("Downloading object names folder")
            os.chdir(download_dir) # Specifying the path where the cloned project needs to be copied
            os.system("git clone https://github.com/CSAILVision/sceneparsing.git") # Cloning
        else:
            print("object names folder has already been downloaded")

        self.data_fraction = 1
        self.data, self.datalen = read_files()
        print("sample data ",self.data["images"][0],self.data["masks"][0])
        print("length of original data ",self.datalen)

        self.datalen_aug = self.get_total_number()

        
        self.name = 'ade20k'
        self.additional_scaling = 1.0
        self.prepare_memory_map()

    def sample_one(self, sample_number = -1, viz = False):
        #print("number of files in dataset ",self.datalen)
        if sample_number==-1:
            sample_number = np.random.randint(0, self.datalen)
        
        #Here mask is a grayscale image but each pixel actually denotes an object class id
        rgb, mask= load(self.data["images"][sample_number], self.data["masks"][sample_number])
        

        if viz:
            interactive_viz(rgb,mask,self.objectnames)

        return rgb, mask

    def one_hot_encode(self,array):
        array = np.array(array, dtype = np.int)
        eye = np.eye(self.n_classes, dtype = np.int)
        ohe = np.stack([eye[i] for i in array], axis=0)
        return ohe

    def get_total_number(self):
        print("In augmentations function")
        count=0
        for idx in range(self.datalen//self.data_fraction):
            
            x, y = self.sample_one(sample_number=idx)
            patches = augmentations.augmented_windowed_patches(x, [augmentations.flip_images], (self.w,self.h))
            count+=len(patches)
            print(count,end='\r')
        print("got total images using augmentation ",count)
        return count
        #sys.exit(0)



    def prepare_memory_map(self): #should only take care of image resize and normalization/ all other image transforms in load
        #for y_train it is assumed that either each pixel stores a regression value
        #or each pixel stores a class label index
        #this information is later converted to one hot encoding using pytorch function
        

        if os.path.exists('x_train'+self.name+'.dat') and os.path.exists('y_train'+self.name+'.dat'):
            print("Data is already memory mapped ")
            self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen_aug, self.h, self.w, 3))
            self.y_train = np.memmap('y_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen_aug, self.h, self.w))
            return

        print("Memory map does not exist, creating one ..")
        self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen_aug, self.h, self.w, 3))
        self.y_train = np.memmap('y_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen_aug, self.h, self.w))

        max_y = self.additional_scaling

        count = 0

        for idx in range(self.datalen//self.data_fraction):
            print(idx,end='\r')

            x, y = self.sample_one(sample_number=idx)

            patches_x = augmentations.augmented_windowed_patches(x, [augmentations.flip_images], (self.w,self.h))
            patches_y = augmentations.augmented_windowed_patches(y, [augmentations.flip_images], (self.w,self.h))

                

            for im_num in range(len(patches_x)):
                
                im_x = cv2.resize(patches_x[im_num],(self.w,self.h))
                im_x = np.array(im_x/255.0,dtype = np.float32)
                self.x_train[count,:] = im_x

                im_y = cv2.resize(patches_y[im_num],(self.w,self.h))
                im_y = np.array(im_y/max_y,dtype = np.float32)
                self.y_train[count,:] = im_y

                count+=1


            
            

            #print("min and max of x ",np.min(x), np.max(x))
            #print("min and max of y ",np.min(y), np.max(y))

    def get_device_batch(self):
        idx = np.random.randint(self.datalen_aug, size=self.batch_size)

        x = torch.from_numpy(self.x_train[idx]).view((self.batch_size, 3, self.w, self.h)).to(self.device)

        #y = torch.LongTensor(self.y_train[idx].astype(np.int64))
        
        y = torch.from_numpy(self.y_train[idx]).view((self.batch_size, self.w, self.h)).to(self.device)
        y = y.to(torch.long)
        #y = F.one_hot(y, num_classes=self.n_classes)
        #y = y.to(torch.long)
        #y = y.view((self.batch_size, 1, self.w, self.h)).to(self.device)

        return x,y



if __name__ == '__main__':
    d = dataloader()
    rgb, mask = d.sample_one(viz=True)

    xt, yt = d.get_device_batch()
    print("sample tensor batch ",type(xt), xt.shape)


