import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud

class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)


def augment(pointcloud):
    norm_pointcloud = Normalize()(pointcloud)
    rot_pointcloud = RandRotation_z()(norm_pointcloud)
    noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)
    return noisy_rot_pointcloud





class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (bs,n,3)

      #do not use batch normalization for now because using batch size 1

      bs = input.size(0)
      #print("TNET input size ",input.size())
      '''
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      '''

      xb = F.relu(self.conv1(input))
      xb = F.relu(self.conv2(xb))
      xb = F.relu(self.conv3(xb))
      #print("TNET xb size ",xb.size())

      #pool = nn.MaxPool1d(xb.size(-1))(xb)
      #flat = nn.Flatten(1)(pool)

      pool = nn.MaxPool1d(xb.size(-1))(xb)
      #print("TNET pool size ",pool.size())
      flat = pool.view(-1, 1024)


      '''
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      '''

      xb = F.relu(self.fc1(flat))
      xb = F.relu(self.fc2(xb))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        #print("transform first layer xb shape ",xb.size())

        #xb = F.relu(self.bn1(self.conv1(xb)))
        xb = F.relu(self.conv1(xb))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        final_local_features = xb

        '''
        xb2 = F.relu(self.bn2(self.conv2(xb)))
        xb2 = self.bn3(self.conv3(xb2))
        '''
        xb2 = F.relu(self.conv2(xb))
        xb2 = self.conv3(xb2)


        xb2 = nn.MaxPool1d(xb.size(-1))(xb2)
        global_features = nn.Flatten(1)(xb2)
        return global_features, final_local_features, matrix3x3, matrix64x64

class PointNet_classif(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        global_features, final_local_features, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(global_features)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)



class convblock(nn.Module):
   def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1088,512,1)
        self.conv2 = nn.Conv1d(512,256,1)
        self.conv3 = nn.Conv1d(256,128,1)
       
   def forward(self, input_tensor):
        x = F.relu(self.conv1(input_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x



class PointNet_seg(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        self.convblock = convblock()

        #conv1d follows channel first input rule
        self.conv1 = nn.Conv1d(1088,128,1)
        self.conv2 = nn.Conv1d(128,classes,1)

        self.bn1 = nn.BatchNorm1d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.classes = classes

    def forward(self, input):
        #(cross checked implementation with below link)
        #https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_part_seg.py

        global_features, final_local_features, matrix3x3, matrix64x64 = self.transform(input)

        '''
        global_features = global_features.view(-1,1,1024)
        global_features = torch.tile(global_features, (1,final_local_features.size(-2),1))
        '''
        global_features = global_features.view(-1,1024)
        global_features = global_features.view(-1, 1024, 1).repeat(1, 1, final_local_features.size(-1))

        
        #print("pointnet seg global features tiled shape ",global_features.size())
        #print("pointnet seg final local features shape ",final_local_features.size())

        concat = torch.cat([global_features, final_local_features], dim=1)
        #print("concat layer shape ",concat.size())
        
        #seg = F.relu(self.bn1(self.conv1(concat)))
        seg = F.relu(self.convblock(concat))

        '''
        seg = F.relu((self.conv2(seg))).transpose(1,2).contiguous() #same as .transpose(2,1)
        print("pointnet seg seg shape ",seg.size())

        seg_l= self.logsoftmax(seg.view(-1,self.classes))
        output = seg_l.view(-1,self.classes,seg.size(1))
        #print("pointnet seg output shape ",output.size())

        print("pointnet seg output shape ",output.shape)
        return output, matrix3x3, matrix64x64
        '''
        seg = F.relu((self.conv2(seg)))
        #print("pointnet seg final layer shape ",seg.size())
        return seg, matrix3x3, matrix64x64




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':
    import os
    import sys
    os.environ['top'] = '../../'
    sys.path.append(os.path.join(os.environ['top']))
    from custom_losses.pytorch import pointnetloss
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on device ",device)

    batch_size = 2
    num_points = 35000 #note - num_points can be anything there is no constraint by the model/ also does not affect number of parameters 
    num_seg_class = 10
    mode = 1
    modes = ["classification","segmentation"]


    if modes[mode]=="classification":
        pointnet = PointNet_classif()
    if modes[mode]=="segmentation":
        pointnet = PointNet_seg(classes = num_seg_class)


    print("Total trainable parameters ",count_parameters(pointnet))


    pointnet.to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)
    pointnet.train()



    #inputs shape - (batch size, num_points, 3)
    
    dummy_input_batch = np.zeros((batch_size,num_points,3))
    for i in range(batch_size):
        dummy_input = np.random.rand(num_points,3)
        dummy_input = augment(dummy_input)
        dummy_input_batch[i,:]=dummy_input

    print("dummy input batch shape ",dummy_input_batch.shape)
    dummy_input = torch.tensor(dummy_input_batch).to(device).float()



    if modes[mode]=="classification":
        dummy_output = np.array([0]*batch_size)
        dummy_output = torch.tensor(dummy_output).to(device)

    if modes[mode]=="segmentation":
        #for each batch, each element out of 2024 is basially an integer denoting the semantic label of the point
        dummy_output = np.random.randint(num_seg_class, size=(batch_size, num_points))
        dummy_output = torch.tensor(dummy_output).to(device).long()

    

    inputs, labels = dummy_input, dummy_output





    optimizer.zero_grad()
    outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
    print("model forward pass success ")
    print("model output argmax shape ",torch.argmax(outputs, dim=1).size())

    loss = pointnetloss.loss(outputs, labels, m3x3, m64x64)
    loss.backward()
    optimizer.step()

    print("sample model forward loss ",loss.item())


