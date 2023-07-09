import os
import copy

import cv2
import math
import json

import numpy as np


#useful for loading your custom image classification/regression datasets
#expected to have used the jsonwriters class from datacreators/utils/jsonlabels.py
#works with json files that stores image names and labels in a compact dictionary


#this function will change based on type of application the dataloader is being used for
def process_dict_ele(folder, elem):
    #fname = '/datasets/behavior_cloning/game1/'+elem["id"]
    fname = folder+elem["id"]

    x = cv2.imread(fname)
    x = cv2.resize(x,(224,224))

    k = elem['keys_pressed']
    y = 0 #8 cardinal directions movement and do nothing

    if k == ["'w'"]:
        y = 0 
    elif k == ["'a'"]:
        y = 1
    elif k == ["'s'"]:
        y = 2
    elif k == ["'d'"]:
        y = 3

    elif "'w'" in k and "'a'" in k:
        y = 4
    elif "'w'" in k and "'d'" in k:
        y = 5
    elif "'s'" in k and "'a'" in k:
        y = 6
    elif "'s'" in k and "'d'" in k:
        y = 7

    elif k == []:
        y = 8

    x = np.array(x/255.0, dtype = np.float32).reshape((3,224,224))
    y = np.array(y, dtype = np.int32)

    return x, y



def viualize_sample(x,y):
    c,h,w = x.shape[-3], x.shape[-2], x.shape[-1]

    if len(x.shape)==5:
        print("sequence data, showing for first batch only ")
        x0 = x[0]
        y0 = y[0]
        for t in range(x0.shape[0]):
            xt = x0[t].reshape((h,w,c))
            cv2.imshow("image t ",xt)
            cv2.waitKey(0)
            yt = y0[t]
            print("recorded agent action encoded ",yt, " max value of image ",np.max(xt))

    if len(x.shape)==4:
        print("IID batch showing entire batch  ")
        x0 = x
        y0 = y
        for t in range(x0.shape[0]):
            xt = x0[t].reshape((h,w,c))
            cv2.imshow("image t ",xt)
            cv2.waitKey(0)
            yt = y0[t]
            print("recorded agent action encoded ",yt, " max value of image ",np.max(xt))





class jsonlabel_loader(object):

    def __init__(self, json_labels_folder, json_labels_file, data_processing_fun):
        #self.filename = json_labels_file
        self.foldername = json_labels_folder
        self.file = json_labels_file
        self.data_processing_fun = data_processing_fun
        
        self.discard_ini_sequence_len = 50
        self.discard_final_sequence_len = 50
        self.skip_sampling = 1

        self.filename = json_labels_folder+json_labels_file
        #filename = "gendata/labels.json"
        with open(self.filename, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
            self.data = json_object

        #print("got data ",self.data)
        #print("data[0] ",self.data[0])
        self.num_labels = len(self.data)
        self.uniform_sampled_label = [1,2,3]


        self.batch_size = 16


    def sample_one(self, idx = 0):
        assert (self.num_labels- self.discard_final_sequence_len -  self.discard_ini_sequence_len) > 16, f"json file does not reference to enough amount of training data"

        if idx==-1:
            id1 = np.random.randint(self.discard_ini_sequence_len, self.num_labels- self.discard_final_sequence_len)
        else:
            id1 = idx

        #print("got randomly sampled id ",id1)
        #x, y = process_dict_ele(self.foldername, self.data[id1])
        x, y = self.data_processing_fun(self.foldername, self.data[id1])

        return x, y

    def try_get_sampled_label(self, sequence_len):
        #required_random_label = np.random.randint(0,4)
        required_random_label = np.random.choice(self.uniform_sampled_label)

        id1 = np.random.randint(self.discard_ini_sequence_len, self.num_labels - (sequence_len*self.skip_sampling) - self.discard_final_sequence_len)
        _, yt = self.data_processing_fun(self.foldername, self.data[id1+(sequence_len*self.skip_sampling)])
        if yt==required_random_label:
            #print("got sampled label ",yt)
            return id1, yt
        while yt!=required_random_label:
            id1 = np.random.randint(self.discard_ini_sequence_len, self.num_labels - (sequence_len*self.skip_sampling) - self.discard_final_sequence_len)
            _, yt = self.data_processing_fun(self.foldername, self.data[id1+(sequence_len*self.skip_sampling)])

            if yt==required_random_label:
                #print("got sampled label ",yt)
                #print("returning try get sample label ",id1)
                return id1, yt
            

    def sample_one_sequence(self, idx = 0, sequence_len = 10):
        if idx==-1:
            if self.uniform_sampled_label!=[]:
                id1, ts = self.try_get_sampled_label(sequence_len)
                #print("using sampled label in sample_one_sequence ",id1)
                #print("targetted sampled end of sequence label ",ts)
            else:
                id1 = np.random.randint(self.discard_ini_sequence_len, self.num_labels - (sequence_len*self.skip_sampling) - self.discard_final_sequence_len)

            



        else:
            id1 = idx
        
        x = []
        y = []

        for n in range(sequence_len):
            #xn, yn = process_dict_ele(self.foldername, self.data[id1+n])
            xn, yn = self.data_processing_fun(self.foldername, self.data[(id1+(n*self.skip_sampling))])
            x.append(xn)
            y.append(yn)

        x = np.stack(x, axis=0)
        y = np.array(y)
        #y = y[-1] #last element needed to get the agent action corresponding to the conditioning of the entire previous time sequence
        #print("sampled sequence y ",y)
        return x,y



    def sample_batch(self, bsize=-1, sequence_len = 1, print_sample = False):
        xb = []
        yb = []
        if bsize==-1:
            bsize = self.batch_size

        for _ in range(bsize):
            if sequence_len>1:
                x,y = self.sample_one_sequence(idx=-1, sequence_len = sequence_len)
            else:
                x,y = self.sample_one(idx=-1)
            
            xb.append(x)
            yb.append(y)

        xb = np.stack(xb, axis=0)
        yb = np.stack(yb, axis=0)

        if print_sample:
            print("xb shape ",xb.shape)
            print("yb shape ",yb.shape)
            print("sample y batch ",yb)
        return xb, yb


if __name__ == '__main__':
    jl = jsonlabel_loader('/datasets/behavior_cloning/game1/','samplelabels.json', process_dict_ele)
    jl.skip_sampling = 10 #can try and experiment with this parameter 
    jl.uniform_sampled_label = [] # put this as empty if skip_sampling>1 otherwise it will take a lot of time findind the right sample
    '''
    x,y = jl.sample_one(idx =-1)
    print("got y ",y)

    xt,yt = jl.sample_one_sequence()
    print("shapes ",xt.shape, yt.shape)
    '''

    xb, yb = jl.sample_batch(bsize = 4, sequence_len = 15)
    print("shapes ",xb.shape, yb.shape)

    viualize_sample(xb,yb) #will visualize the sequence

    xb, yb = jl.sample_batch(bsize = 4, sequence_len = 1) #passing sequence len =1 means iid sampling without any sequence assumption
    print("shapes ",xb.shape, yb.shape)
    print("max of xb ",np.max(xb))

    #viualize_sample(xb,yb) #visualize entire batch (when not a sequence)