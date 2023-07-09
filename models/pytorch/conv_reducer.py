# A definition of moderately heavy convolution block that takes in input a fixed size image and maps it to a fixed size 1D vector
# Some parts where the tensor is reshaped using view() is hardcoded for input of size 128x128

import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class MLP(nn.Module): #pass hidden size as a list of widths of each nn layer in the series
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)

class convblock(nn.Module):
    def __init__(self, in_channel = 3, channel_size = 32):
        super(convblock, self).__init__()
        self.in_channel = in_channel
        self.channel_size = channel_size
        self.proj = nn.Conv2d(self.in_channel, self.channel_size, 3, padding = 1, stride = 2)
        '''
        self.block = nn.Sequential( nn.Conv2d(self.in_channel, self.channel_size, 3, padding = 1, dilation =1) , nn.ReLU(inplace = True),
                                    nn.Conv2d(self.channel_size, self.channel_size, 3, padding = 1, dilation = 2) , nn.ReLU(inplace = True),
                                    nn.Conv2d(self.channel_size, self.channel_size//2, 3, padding = 1, stride = 2, dilation =3) , nn.ReLU(inplace = True),
                                    nn.Conv2d(self.channel_size//2, self.channel_size//2, 1, padding = 1) , nn.ReLU(inplace = True),
                                    nn.Conv2d(self.channel_size//2, self.channel_size, 3, padding = 1) , nn.ReLU(inplace = True),
                                    )
        '''

        self.block = nn.Sequential( nn.Conv2d(self.in_channel, self.channel_size, 3, padding = 1) , nn.ReLU(inplace = True),
                                    nn.Conv2d(self.channel_size, self.channel_size, 3, padding = 1) , nn.ReLU(inplace = True),
                                    nn.Conv2d(self.channel_size, self.channel_size//2, 3, padding = 1, stride = 2) , nn.ReLU(inplace = True),
                                    nn.Conv2d(self.channel_size//2, self.channel_size//2, 1, padding = 1) , nn.ReLU(inplace = True),
                                    nn.Conv2d(self.channel_size//2, self.channel_size, 3, padding = 0) , nn.ReLU(inplace = True),
                                    )
        #self.ln = nn.LayerNorm(self.channel_size)
    def forward(self, x):
        y = self.proj(x)
        y = nn.ReLU()(y)
        x = self.block(x)
        #print("convblock forward shapes ",x.shape,y.shape)
        y = y+x

        return x

class convblock2(nn.Module):
    def __init__(self, in_channel = 3, channel_size = 32):
        super(convblock2, self).__init__()
        self.in_channel = in_channel
        self.channel_size = channel_size
        self.proj = nn.Conv2d(self.in_channel, self.channel_size, 3, padding = 1)
        self.block = nn.Sequential( nn.Conv2d(self.in_channel, self.in_channel//2, 3, padding = 1) , nn.ReLU(inplace = True),
                                    nn.Conv2d(self.in_channel//2, self.channel_size, 3, padding = 1) , nn.ReLU(inplace = True),
                                    )
        #self.ln = nn.LayerNorm(self.channel_size)
    def forward(self, x):
        y = self.proj(x)
        y = nn.ReLU()(y)
        x = self.block(x)
        #print("convblock2 forward shapes ",x.shape,y.shape)
        y = y+x



        return x

class conv1dblock(nn.Module):
    def __init__(self, in_channel = 3, channel_size = 32):
        super(conv1dblock, self).__init__()
        self.in_channel = in_channel
        self.channel_size = channel_size
        
        self.block = nn.Sequential( nn.Conv1d(self.in_channel, self.in_channel//2, 3, padding = 0) , nn.ReLU(inplace = True),
                                    nn.Conv1d(self.in_channel//2, self.in_channel//4, 3, padding = 0) , nn.ReLU(inplace = True),
                                    nn.Conv1d(self.in_channel//4, self.channel_size, 3, padding = 0) , nn.ReLU(inplace = True),
                                    )
        #self.ln = nn.LayerNorm(self.channel_size)
    def forward(self, x):
        
        x = self.block(x)

        return x


class Encoder(nn.Module):
    def __init__(self, shape = (128,128,3), n_hidden = 64):
        super(Encoder, self).__init__()
        
        self.h,self.w,self.c = shape[0],shape[1],shape[2]
        self.eh, self.ew, self.ec = self.h//16, self.w//16, 8

        self.mlps1, self.mlps2, self.mlps3 = self.eh*self.ew*self.ec, self.eh*self.ew*self.ec//2, self.eh*self.ew*self.ec//4
        #remember for conv the order of arguments are - numfilters_in, num_filters_out, kernel_size

        '''
        self.encode = nn.Sequential(convblock(in_channel = 3, channel_size = 32), nn.BatchNorm2d(32),
                                    convblock(in_channel = 32, channel_size = 64), nn.BatchNorm2d(64),
                                    convblock(in_channel = 64, channel_size = 64), nn.BatchNorm2d(64),
                                    convblock(in_channel = 64, channel_size = 64), nn.BatchNorm2d(64), 
                                   )
        '''

        self.encode = nn.Sequential(convblock(in_channel = self.c, channel_size = 32),
                                    convblock(in_channel = 32, channel_size = 64),
                                    convblock(in_channel = 64, channel_size = 64),
                                    convblock(in_channel = 64, channel_size = 128), 
                                    convblock2(in_channel = 128, channel_size = 64), 
                                    convblock2(in_channel = 64, channel_size = 32), 
                                    convblock2(in_channel = 32, channel_size = 16), 
                                    convblock2(in_channel = 16, channel_size = 8), 
                                   )

        self.encode_1d = conv1dblock(in_channel=64, channel_size = 8)
        self.flatten = Flatten()

        #run forward the network to get to know the first value in the list passed to MLP (here 344)
        #self.ffblock = MLP([self.mlps1, self.mlps2, self.mlps3, n_hidden])
        self.ffblock = MLP([self.mlps1, n_hidden])

    def forward(self, x, y = None):
        '''
        x = self.encode(x)
        #print("shape after encode ",x.shape)
        eh, ew, ec = x.shape[-1], x.shape[-2], x.shape[-3]
        x = x.view(-1,ec,eh*ew)
        x = self.encode_1d(x)
        #print("shape before flatten ",x.shape)
        x = self.flatten(x)
        #print("shape before ffblock ",x.shape)
        x = self.ffblock(x)
        return x
        '''

        x = self.encode(x)
        #print("shape after encode ",x.shape)
        #print("shape before flatten ",x.shape)
        x = self.flatten(x)
        #print("shape before ffblock ",x.shape)
        x = self.ffblock(x)
        return x

if __name__ == '__main__':

    #Able to reduce 128x128x3 image to a 64 dimensional 1d vector
    #Can customize Encoder for greater reductions
    #Using residual convolution blocks
    #Using dilation and striding instead of maxpool2d
    #Tensor obtained after flatten is further passed through conv1d block to further reduce dimensions, before finally passing to fully connected block

    bsize = 4
    imshape = (32,32,1)

    cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    e = Encoder(shape = imshape).to(device)


    print(sum(p.numel() for p in e.parameters())/1e6, 'M parameters')
    t = torch.randn((bsize, imshape[2], imshape[0],imshape[1])).to(device)

    out = e(t)
    print("model out shape ",out.shape)