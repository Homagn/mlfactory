###############################
# download tum_rgbd dataset example
# go to https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg1_xyz
# pick any one of the samples then do these (suppose I do these in the /datasets folder)

# wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
# tar zxvf rgbd_dataset_freiburg2_desk.tgz 



import os
import sys

import numpy as np
import cv2


import re
import copy
from scipy.spatial.transform import Rotation as R

import torch



cimportpath = os.getcwd()
if cimportpath[cimportpath.rfind("/")+1:]=="dataloaders": #if this module is called from dataloaders code
    os.environ['top'] = '../'
    os.environ['applications'] = '../applications'
    sys.path.append(os.path.join(os.environ['top']))
    sys.path.append(os.path.join(os.environ['applications']))

if cimportpath[cimportpath.rfind("/")+1:]=="examples": #if this module is called from dataloaders code
    os.environ['top'] = '../'
    os.environ['applications'] = '../applications'
    sys.path.append(os.path.join(os.environ['top']))
    sys.path.append(os.path.join(os.environ['applications']))


from applications.superglue_inference import match_pair

def read_files(path, ftype = "rgb"):


    #path = "/datasets/rgbd_dataset_freiburg2_desk/"
    print("path ",path)
    filelist = []

    indexfile = ""

    image_names = []
    timestamps = []
    trajectories = []


    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ftype+".txt"):
                indexfile = os.path.join(root, file)
                print("index file ",indexfile)
                

                with open(indexfile) as f:
                    lines = f.readlines()

                for l in lines:
                    #l = l[:l.rfind('n')]
                    #print("l ",l)
                    
                    tstamp = l[:l.find(" ")]
                    try:
                        timestamps.append(float(tstamp))
                        #print("got tstamp ",tstamp)

                        l = l[l.find(" ")+1:-1]
                        #print("root ",root)
                        #print("l ",l)
                        if ftype=="rgb" or ftype=="depth":
                            image_names.append(os.path.join(root, l))  
                        if ftype=="groundtruth":
                            vals = l.split(' ')
                            trajectories.append([float(v) for v in vals])
                            #print("vals ",vals)
                            #trajectories.append(float())  

                    except:
                        print("skipping header")            
                    
    return timestamps, image_names, trajectories


def align_files(ts1, ts2):
    #ts1 is the much higher frequency time stamps for ground truth trajectory
    #ts2 is the lower freq time stamps for the rgb and depth trajectory
    i  = 0
    j  = 0
    id1 = []
    id2 = []
    prev_diff = (ts1[i] - ts2[j])**2
    i+=1
    while (i<len(ts1)-1 and j<len(ts2)-1):
        diff = (ts1[i] - ts2[j])**2
        if diff<prev_diff:
            i+=1
        else:
            id1.append(i-1)
            id2.append(j)
            #print("i-1, j ",ts1[i-1],ts2[j])
            j+=1
            i+=1
        prev_diff = (ts1[i-1] - ts2[j])**2


    return id1, id2


class loader(object):
    def __init__(self, path = "/datasets/rgbd_dataset_freiburg2_desk/", align_paths = {"rgbfiles":[], "depthfiles":[], "traj": []} ):

        self.path = path
        self.align_paths = align_paths

        ts1, _, tr1 = read_files(self.path, ftype = "groundtruth")
        ts2, im2, _ = read_files(self.path, ftype = "depth")
        ts3, im3, _ = read_files(self.path, ftype = "rgb")

        i1, i2 = align_files(ts1,ts2)


        #print("aligned indices ",len(i1), len(i2), len(im2), len(im3) )

        i2_, i3 = align_files(ts1,ts3)
        #print("aligned indices ",len(i2_), len(i3))

        self.align_paths["rgbfiles"].extend(  [im3[k] for k in i3] )
        self.align_paths["depthfiles"].extend([im2[k] for k in i2] )
        self.align_paths["traj"].extend( [tr1[k] for k in i1] )

        #print("sample time stamps ",ts1[i1[0]], ts2[i2[0]], ts3[i3[0]])


    def sample(self, index = -1, viz = True):
        if index==-1:
            index = np.random.choice(len(self.align_paths["rgbfiles"]))

        rgbfile = self.align_paths["rgbfiles"][index]
        depthfile = self.align_paths["depthfiles"][index]
        traj = self.align_paths["traj"][index]

        if viz:
            cv2.imshow("rgb ",cv2.imread(rgbfile))
            cv2.imshow("depth ",cv2.imread(depthfile))

            cv2.waitKey(0)
            print("got sample trajectory ",traj)

        rgb = cv2.imread(rgbfile)
        depth = cv2.imread(depthfile,0)

        return rgb, depth, traj



class visual_odometry_loader(object):
    def __init__(self, path = "/datasets/rgbd_dataset_freiburg2_desk/", align_paths = {"rgbfiles":[], "depthfiles":[], "traj": []}, datasavepath = "/datasets/"):
        self.tl = loader(path = path, align_paths = align_paths)
        self.m = match_pair.matcher()
        self.datasavepath = datasavepath
        self.datalen = 2940
        self.batch_size = 16
        self.all_labels = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        #used for selecting and normalizing the labels
        self.bounds = [(-0.0042,0.0054), (-0.0023, 0.0074), (-0.0019,0.002), (-0.0016,0.0018),(-0.0019,0.0018),(-0.0043,0.0004), (-1.0,1.0) ]
        #used to skip number of frames in the image sequence while sampling x_train
        self.skip = 1
        #select at most this number of superglue matches
        self.num_matches = 200

    def sample(self, index = -1, viz = False):
        if index== -1:
            index = np.random.choice(2940)

        def convert_close(diff):
            conv = []
            for d in diff:
                
                if 360-np.abs(d) < np.abs(d):
                    conv.append( -np.sign(d)*(360-np.abs(d) ) )
                else:
                    conv.append(d)
                
                #conv.append(d)
            return conv

        def conjugate(q):
            quat = np.array(q.as_quat())
            return R.from_quat([-quat[0], -quat[1], -quat[2], quat[3]])



        
        skip = self.skip
        num_matches = self.num_matches
        norm_t = [np.array([0.0,0.0,0.0]) , 1.0]
        norm_r = [np.array([0.0,0.0,0.0]) , 1.0]

        rg1,d1,t1 = self.tl.sample(index, viz = viz)
        rg2,d2,t2 = self.tl.sample(index+skip, viz = viz)
        rg3,d3,t3 = self.tl.sample(index+skip+skip, viz = viz)

        i1,i2 = self.m.prepare_images(rg1,rg2)
        fm1,fm2 = self.m.match(i1,i2, viz = viz)

        i22,i3 = self.m.prepare_images(rg2,rg3)
        fm22,fm3 = self.m.match(i22,i3, viz = viz)

        #print("fm1 fm2 shapes ",fm1.shape, fm2.shape)
        #print("fm22 fm3 shapes ",fm22.shape, fm3.shape)
        corres = np.vstack((fm1.reshape(2,-1)[:,:num_matches], fm2.reshape(2,-1)[:,:num_matches], fm22.reshape(2,-1)[:,:num_matches], fm3.reshape(2,-1)[:,:num_matches] ))
        print("corres shape ",corres.shape)
        #return the matching point pairs (2,2,200) and the trajectory pairs (2,6)

        
        r1 = R.from_quat(t1[3:]) #quat in format x,y,z,w
        #print("euler ",r1.as_euler('zxy', degrees = True))
        r2 = R.from_quat(t2[3:])
        #print("euler ",r2.as_euler('zxy', degrees = True))
        r3 = R.from_quat(t3[3:])
        #print("euler ",r3.as_euler('zxy', degrees = True))

        diff_trans =  np.array(t1[:3]) - np.array(t3[:3]) 
        diff_rpy = ( convert_close ( (np.array(r1.as_euler('zxy', degrees = True)) - np.array(r3.as_euler('zxy', degrees = True)) ) ) +norm_r[0] ) /norm_r[1]
        #diff_quat = np.array(t1[3:]) - np.array(t3[3:]) 
        diff_quat = r1*conjugate(r3)
        diff_quat = np.array(diff_quat.as_quat())

        print("diff trans ",diff_trans)
        print("diff_rpy ",diff_rpy)
        print("diff_quat ",diff_quat)

        return corres,np.concatenate((diff_trans,diff_quat))

        #return corres as input and diff_trans and diff_rpy as labels

    def prepare_files(self):
        if os.path.exists(self.datasavepath + "x_train/") and os.path.exists(self.datasavepath + "y_train/"):
            return

        else:
            os.mkdir(self.datasavepath + "x_train")
            os.mkdir(self.datasavepath + "y_train")
            print("Memory map does not exist, creating one ..")
            


            for idx in range(self.datalen):
                #print(idx,end='\r')
                print("creating sample number ",idx)

                x, y = self.sample(index=idx)
                self.all_labels.append(y)

                

                np.save(self.datasavepath + "x_train/"+str(idx)+".npy", x)
                np.save(self.datasavepath + "y_train/"+str(idx)+".npy", y)

            print("Memory map created ")
            self.all_labels = np.array(self.all_labels)

    
    def validate(self, label):
        a,b,c = label[0],label[1],label[2]
        x,y,z,w = label[3],label[4],label[5],label[6]

        

        if a<self.bounds[0][0] or a>self.bounds[0][1] or b<self.bounds[1][0] or b>self.bounds[1][1] or c<self.bounds[2][0] or c>self.bounds[2][1]:
            return False
        if x<self.bounds[3][0] or x>self.bounds[3][1] or y<self.bounds[4][0] or y>self.bounds[4][1] or z<self.bounds[5][0] or z>self.bounds[5][1] or w<self.bounds[6][0] or w>self.bounds[6][1]:
            return False
        
        return True


    def get_stats(self):
        all_labels = []
        for i in range(self.datalen):
            y_i = np.load(self.datasavepath+"y_train/"+str(i)+".npy")
            all_labels.append(y_i)
        all_labels = np.array(all_labels)
        print("stats ", all_labels[:,0].shape)
        for c in range(7):
            #print(np.std(all_labels[:,c]),np.median(all_labels[:,c]),np.min(all_labels[:,c]),np.max(all_labels[:,c]))
            print(np.percentile(all_labels[:,c], 25), np.percentile(all_labels[:,c], 75))


    def normalize(self, xt,yt):
        xt = xt/640.0
        for c in range(7):
            yt[c] = (yt[c]-self.bounds[c][0])/(self.bounds[c][1]-self.bounds[c][0])
        xt = np.array(xt,dtype = np.float32)
        yt = np.array(yt,dtype = np.float32)
        return xt,yt


    def get_device_batch(self):
        #idx = np.random.randint(self.datalen, size=self.batch_size)

        x = np.zeros((self.batch_size, 8, 200), dtype = np.float32)
        y = np.zeros((self.batch_size, 7), dtype = np.float32)

        count = 0
        while count<self.batch_size:
            i = np.random.randint(self.datalen)

            x_i = np.load(self.datasavepath+"x_train/"+str(i)+".npy")
            y_i = np.load(self.datasavepath+"y_train/"+str(i)+".npy")


            isvalid = self.validate(y_i) # make sure the deviation in the data is within acceptable limits

            if isvalid:
                x_i,y_i = self.normalize(x_i,y_i)
                x[count] = x_i
                y[count] = y_i

                count+=1

        x_ = torch.from_numpy(x).view((self.batch_size, 8, 200)).float().to(self.device)
        y_ = torch.from_numpy(y).view((self.batch_size, 7 )).float().to(self.device)

        #x_ = torch.tensor(x_, requires_grad=True)
        #y_ = torch.tensor(y_, requires_grad=True)

        return x_,y_







if __name__ == '__main__':
    
    vol = visual_odometry_loader()
    vol.prepare_files()
    vol.get_stats()
    xt,yt = vol.get_device_batch()
    print("sample yt ",yt)
    print("sample xt ",xt)
    vol.get_stats()

    '''

    all_changes = vol.all_labels
    print("stats", all_changes[:,0].shape)
    print(np.max(all_changes[:,0]), np.min(all_changes[:,0]))
    print(np.max(all_changes[:,1]), np.min(all_changes[:,1]))
    print(np.max(all_changes[:,2]), np.min(all_changes[:,2]))
    print(np.max(all_changes[:,3]), np.min(all_changes[:,3]))
    print(np.max(all_changes[:,4]), np.min(all_changes[:,4]))
    print(np.max(all_changes[:,5]), np.min(all_changes[:,5]))
    '''
        

    '''
    l = loader()
    r1,d1,t1 = l.sample(20)
    r2,d2,t2 = l.sample(21)

    m = match_pair.matcher()

    m.match(np.asarray(d1),np.asarray(d2))
    '''

    # gdown --id 1RU5jIDkvaXE_h0fEew-xYCiJs87QDMEG