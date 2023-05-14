import os
import sys

import numpy as np
import cv2

import torch

os.environ['top'] = '../'
sys.path.append(os.path.join(os.environ['top']))



from scipy import ndimage

import re
import copy


def edges(d):
    #single derivative kind of does the job, adding double derivative does even better job

    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dx2 = ndimage.sobel(dx, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    dy2 = ndimage.sobel(dy, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)+ np.abs(dx2) + np.abs(dy2)

# This is special function used for reading NYU pgm format
# as it is written in big endian byte order.
def read_nyu_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out


def read_files(dloc):
    #path = "/datasets/nyuv2/"
    path = dloc
    #scenarios = ["basements","bedrooms_part1","cafe","dining_rooms_part1","home_offices","kitchens_part1","living_rooms_part2","offices_part1"]
    scenarios = ["basements"]
    

    data = {
        "image": [],
        "depth": []
    }

    for s in scenarios:
        path = dloc+s
        print("path ",path)
        filelist = []

        indexfile = ""

        image_names = []
        depth_names = []

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".txt"):
                    indexfile = os.path.join(root, file)
                    with open(indexfile) as f:
                        lines = f.readlines()

                    for l in lines:
                        l = l[:l.rfind('n')]
                        #print(l)
                        if l.endswith('.pgm'):
                            depth_names.append(os.path.join(root, l))
                        if l.endswith('.ppm'):
                            image_names.append(os.path.join(root, l))

                        #sometimes there are unequal number of rgb and depth images
                        if len(image_names)>len(depth_names)+1:
                            image_names.pop()
                        if len(depth_names)>len(image_names)+1:
                            depth_names.pop()
                        

                    break
                
                #filelist.append(os.path.join(root, file))

        '''
        if len(depth_names)<len(image_names):
            image_names = image_names[:len(depth_names)]
        if len(image_names)<len(depth_names):
            depth_names = depth_names[:len(image_names)]
        '''

        image_names.sort(key=lambda x: float( x [ x.find('-')+1:x.rfind('-') ] ) , reverse=True)
        depth_names.sort(key=lambda x: float( x [ x.find('-')+1:x.rfind('-') ] ) ,reverse = True)

        #print("image_names ",image_names[0:5])
        #print("depth names ",depth_names[0:5])
        #sys.exit(0)

        '''
        image_names =  [x for x in filelist if x.endswith(".ppm")]
        depth_names =  [x for x in filelist if x.endswith(".pgm")]
        image_names.sort(key=lambda x: float( x [ x.find('-')+1:x.rfind('-') ] ) )
        depth_names.sort(key=lambda x: float( x [ x.find('-')+1:x.rfind('-') ] ) )
        
        '''


        data["image"]+=image_names
        data["depth"]+=depth_names
        


    print("sample path names ..................")
    print("image ",data["image"][0])
    print("depth ",data["depth"][0])

    '''
    cv2.imshow("sample rgb ", cv2.imread(data["image"][0]))
    cv2.waitKey(0)
    cv2.imshow("sample depth ",read_nyu_pgm(data["depth"][0]))
    cv2.waitKey(0)
    '''
    return data, len(data["image"])-1
    



def load(image_path, depth_path):
    #print("image path supplied ",image_path)
    #print("depth path supplied ",depth_path)

    image = cv2.imread(image_path)
    depth = read_nyu_pgm(depth_path)

    return image, depth, np.max(depth)






class dataloader(object):
    def __init__(self, datalocation, numpy_memmap = False):
        self.batch_size = 8
        self.w = 640
        self.h = 480

        #sometimes model might output a different size image
        self.w_target = 160
        self.h_target = 120

        self.nc = 3
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.additional_scaling = 65000.0 #depends on the camera used to collect depth

        self.datalocation = datalocation
        self.numpy_memmap = numpy_memmap

        self.data, self.datalen = read_files(self.datalocation)
        print("Number of examples in the dataset ",self.datalen)
        self.data_fraction = 2
        self.datalen = self.datalen//self.data_fraction
        self.name = '_nyuv2'

        if self.numpy_memmap:
            self.prepare_memory_map()
        else:
            self.prepare_files()


    def prepare_files(self):
        if os.path.exists("x_train/") and os.path.exists("y_train/"):
            return

        else:
            os.mkdir("x_train")
            os.mkdir("y_train")
            print("Memory map does not exist, creating one ..")
            

            #max_y = self.additional_scaling
            #actually the max will be around 65000, but not normalizing y because last layer relu is used and also if that 
            #entire range is squashed between 0 and 1 the loss propagated will be very low and model wont train properly
            #models for regression generally train well with huge ranges and high resolution
            max_y = 1.0 

            for idx in range(self.datalen):
                #print(idx,end='\r')
                print("creating sample number ",idx)

                x, y = self.sample_one(sample_number=idx)

                x = cv2.resize(x,(self.w,self.h))
                if self.nc==1:
                    #change to grayscale because depth estimation should not be biased on color but just the overall intensity
                    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

                x = np.array(x/255.0,dtype = np.float32)
                

                y = cv2.resize(y,(self.w_target,self.h_target))
                #y = np.array(y,dtype = np.float32) #lets try not normalizing the depth map and put relu in the last layer via loss
                y = np.array(y/max_y,dtype = np.float32) #lets try not normalizing the depth map and put relu in the last layer via loss
                #print("max image y ",np.max(y))
                y = np.reshape(y,(y.shape[0],y.shape[1],1))

                #self.x_train[idx,:] = x
                #self.y_train[idx,:] = y

                np.save("x_train/"+str(idx)+".npy", x)
                np.save("y_train/"+str(idx)+".npy", y)

            print("Memory map created ")









    def prepare_memory_map(self): #should only take care of image resize, graying and normalization/ all other image transforms in load
        if os.path.exists('x_train'+self.name+'.dat') and os.path.exists('y_train'+self.name+'.dat'):
            print("Data is already memory mapped ")
            if self.nc>1:
                self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen, self.h, self.w, self.nc))
            else:
                self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen, self.h, self.w))
            self.y_train = np.memmap('y_train'+self.name+'.dat', dtype=np.float32, mode='r', shape=(self.datalen, self.h_target, self.w_target))
            return

        print("Memory map does not exist, creating one ..")
        if self.nc>1:
            self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen, self.h, self.w, self.nc))
        else:
            self.x_train = np.memmap('x_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen, self.h, self.w))
        self.y_train = np.memmap('y_train'+self.name+'.dat', dtype=np.float32, mode='w+', shape=(self.datalen, self.h_target, self.w_target))

        #max_y = self.additional_scaling
        #actually the max will be around 65000, but not normalizing y because last layer relu is used and also if that 
        #entire range is squashed between 0 and 1 the loss propagated will be very low and model wont train properly
        #models for regression generally train well with huge ranges and high resolution
        max_y = 1.0 

        for idx in range(self.datalen):
            print(idx,end='\r')

            x, y = self.sample_one(sample_number=idx)

            x = cv2.resize(x,(self.w,self.h))
            if self.nc==1:
                #change to grayscale because depth estimation should not be biased on color but just the overall intensity
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

            x = np.array(x/255.0,dtype = np.float32)
            

            y = cv2.resize(y,(self.w_target,self.h_target))
            #y = np.array(y,dtype = np.float32) #lets try not normalizing the depth map and put relu in the last layer via loss
            y = np.array(y/max_y,dtype = np.float32) #lets try not normalizing the depth map and put relu in the last layer via loss
            #print("max image y ",np.max(y))

            self.x_train[idx,:] = x
            self.y_train[idx,:] = y

        print("Memory map created ")


    def sample_one(self, sample_number = -1):
        #print("number of files in dataset ",self.datalen)
        if sample_number==-1:
            sample_number = np.random.randint(0, self.datalen)
        #print("viewing sample number ",sample_number)
        isvalid = False
        sn = copy.copy(sample_number)
        
        while not isvalid:
            try:
                rgb, d, md = load(self.data["image"][sn], self.data["depth"][sn])
                test = cv2.resize(rgb,(rgb.shape[1],rgb.shape[0]))

                isvalid = True
            except:
                print("This was a faulty file ",self.data["image"][sample_number])
                random = np.random.randint(0, self.datalen)
                rgb, d, md = load(self.data["image"][random], self.data["depth"][random])
                sn = random
        

        

        return rgb, d

    def get_device_batch(self):
        idx = np.random.randint(self.datalen, size=self.batch_size)


        if self.numpy_memmap:
            x = torch.from_numpy(self.x_train[idx]).view((self.batch_size, self.nc, self.h, self.w)).to(self.device)
            y = torch.from_numpy(self.y_train[idx]).view((self.batch_size, 1, self.h_target, self.w_target)).to(self.device)



        else:
            x = np.zeros((self.batch_size, self.h, self.w, self.nc), dtype = np.float32)
            y = np.zeros((self.batch_size, self.h_target, self.w_target, 1), dtype = np.float32)

            count = 0
            for i in idx:
                x_i = np.load("x_train/"+str(i)+".npy")
                y_i = np.load("y_train/"+str(i)+".npy")

                x[count] = x_i
                y[count] = y_i

                count+=1

            x = torch.from_numpy(x).view((self.batch_size, self.nc, self.h, self.w)).to(self.device)
            y = torch.from_numpy(y).view((self.batch_size, 1, self.h_target, self.w_target)).to(self.device)


        return x,y




if __name__ == '__main__':

    dt = dataloader()
    rgb, d = dt.sample_one()
    from visualizers import pointcloud

    print("got maximum of depth map from sampled map ",np.max(d))

    #from nyu toolbox download/ camera_params.m
    #multiply the value found in fx_d and fy_d by 2
    #if in the pointcloud it seems that the object walls and edges are leaning towards you
    #then reduce focal lenghts
    #otherwise increase focal lengths to achieve perfect pointcloud
    #also youll observe a cone spread type of artifact in front of the pointcloud if focal lenghts are incorrect
    nyu_camera_params = {
                        "fx": 2*582.62,
                        "fy": 2*582.69,
                        "centerX": 313.04,
                        "centerY": 238.44,
                        "scalingFactor": 5000 
                        }

    

    # 1. if you want to check a default training output
    pointcloud.show_pcd_from_rgbd(rgb, d, nyu_camera_params)
    
    # OR
    
    
    #2. if you want to test on some image
    '''
    test_idx = 90
    save_pcd_loc = "sample_pcds/"+str(test_idx)+"_"
    rgb_test = cv2.imread("../applications/mono_depth/predictions/past/input_"+str(test_idx)+".png")
    d_test_target = cv2.imread("../applications/mono_depth/predictions/past/target_"+str(test_idx)+".png",0)
    d_test_pred = cv2.imread("../applications/mono_depth/predictions/past/"+str(test_idx)+".png",0)
    rgb_test = cv2.resize(rgb_test,(160,120))
    #if sometimes depth map has a lot of edges
    #d_test[edges(d_test) > 0.5] = 0.0  # Hide depth edges   
    
    #the output depth map for feature pyramid model is scaled down 4 times
    nyu_camera_params = pointcloud.scale_camera_params(4,4,nyu_camera_params)

    cv2.imwrite(save_pcd_loc+"_input_image"+".png",rgb_test)
    pointcloud.show_pcd_from_rgbd(rgb_test, d_test_target, nyu_camera_params, save_pcd_loc+"target.pcd")
    pointcloud.show_pcd_from_rgbd(rgb_test, d_test_pred, nyu_camera_params, save_pcd_loc+"pred.pcd")
    '''




    #test sample tensor batch
    xt, yt = dt.get_device_batch()
    print("sample tensor batch ",type(xt), xt.shape)


