# A definition of moderately heavy convolution block that takes in input a 1d vector and upscales it to a larger 2D image like vector
# Some parts where the tensor is reshaped using view() is hardcoded for input of size 128x128

import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict


#probably a better alternative to upsampling2d 
#https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/master/models.py#L62C1-L62C1

class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        #print("output shape of subpixel conv ",output.shape)
        return output


class MLP(nn.Module): #pass hidden size as a list of widths of each nn layer in the series
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                #q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU()))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)

class convblock(nn.Module):
    def __init__(self, in_channel = 3, channel_size = 32, last_block = False):
        super(convblock, self).__init__()
        self.in_channel = in_channel
        self.channel_size = channel_size

        if last_block:
            self.last_act = nn.Identity()
        else:
            self.last_act = nn.ReLU(inplace = True)

        if channel_size!=in_channel:
            self.proj = nn.Sequential( nn.Conv2d(self.in_channel, self.channel_size, 3, padding = 1) , nn.ReLU(),
                                        SubPixelConvolutionalBlock(kernel_size = 3, n_channels = self.channel_size, scaling_factor = 2),
                                        )
        else:

            self.proj = SubPixelConvolutionalBlock(kernel_size = 3, n_channels = self.channel_size, scaling_factor = 2)

        '''
        self.block = nn.Sequential( nn.Conv2d(self.in_channel, self.channel_size, 3, padding = 1, dilation =1) , nn.ReLU(),
                                    nn.Conv2d(self.channel_size, self.channel_size, 3, padding = 1, dilation = 2) , nn.ReLU(),
                                    nn.Conv2d(self.channel_size, self.channel_size//2, 3, padding = 1) , nn.ReLU(),
                                    SubPixelConvolutionalBlock(kernel_size = 3, n_channels = self.channel_size//2, scaling_factor = 2), 
                                    nn.Conv2d(self.channel_size//2, self.channel_size//2, 1, padding = 1) , nn.ReLU(),
                                    nn.Conv2d(self.channel_size//2, self.channel_size, 3, padding = 2) , nn.ReLU(),
                                    )
        '''
        self.block = nn.Sequential( nn.Conv2d(self.in_channel, self.channel_size, 3, padding = 1) , nn.ReLU(),
                                    nn.Conv2d(self.channel_size, self.channel_size, 3, padding = 1) , nn.ReLU(),
                                    nn.Conv2d(self.channel_size, self.channel_size//2, 3, padding = 0) , nn.ReLU(),
                                    SubPixelConvolutionalBlock(kernel_size = 3, n_channels = self.channel_size//2, scaling_factor = 2), 
                                    nn.Conv2d(self.channel_size//2, self.channel_size//2, 1, padding = 1) , nn.ReLU(),
                                    nn.Conv2d(self.channel_size//2, self.channel_size, 3, padding = 2) , nn.ReLU(),
                                    )
        #self.ln = nn.LayerNorm(self.channel_size)
    def forward(self, x):
        y = self.proj(x)
        x = self.block(x)
        y = y+x

        return x

class conv1dblock(nn.Module):
    def __init__(self, in_channel = 3, channel_size = 32):
        super(conv1dblock, self).__init__()
        self.in_channel = in_channel
        self.channel_size = channel_size
        
        self.block = nn.Sequential( nn.Conv1d(self.in_channel, self.in_channel*2, 3, padding = 2) , nn.ReLU(),
                                    nn.Conv1d(self.in_channel*2, self.in_channel*4, 3, padding = 2) , nn.ReLU(),
                                    nn.Conv1d(self.in_channel*4, self.channel_size, 3, padding = 2) , nn.ReLU(),
                                    )
        #self.ln = nn.LayerNorm(self.channel_size)
    def forward(self, x):
        
        x = self.block(x)

        return x

class convblock2(nn.Module):
    def __init__(self, in_channel = 3, channel_size = 32):
        super(convblock2, self).__init__()
        self.in_channel = in_channel
        self.channel_size = channel_size
        self.proj = nn.Conv2d(self.in_channel, self.channel_size, 3, padding = 1)
        self.block = nn.Sequential( nn.Conv2d(self.in_channel, self.in_channel//2, 3, padding = 1) , nn.ReLU(),
                                    nn.Conv2d(self.in_channel//2, self.channel_size, 3, padding = 1) , nn.ReLU(),
                                    )
        #self.ln = nn.LayerNorm(self.channel_size)
    def forward(self, x):
        y = self.proj(x)
        y = nn.ReLU()(y)
        x = self.block(x)
        #print("convblock2 forward shapes ",x.shape,y.shape)
        y = y+x



        return x

class Decoder(nn.Module):
    def __init__(self, shape = (128,128,3), n_hidden = 64):
        super(Decoder, self).__init__()

        self.h,self.w,self.c = shape[0],shape[1],shape[2]
        self.eh, self.ew, self.ec = self.h//16, self.w//16, 8
        if self.ew<=2 or self.eh<=2:
            self.eh, self.ew, self.ec = 8, 8, 8

        self.mlps1, self.mlps2, self.mlps3 = self.eh*self.ew*self.ec, self.eh*self.ew*self.ec//2, self.eh*self.ew*self.ec//4
        #remember for conv the order of arguments are - numfilters_in, num_filters_out, kernel_size

        
        '''
        self.decode = nn.Sequential(convblock(in_channel = 64, channel_size = 64), nn.BatchNorm2d(64),
                                    convblock(in_channel = 64, channel_size = 64), nn.BatchNorm2d(64),
                                    convblock(in_channel = 64, channel_size = 32), nn.BatchNorm2d(32),
                                    convblock(in_channel = 32, channel_size = 3, last_block = True), 
                                   )
        '''

        self.decode = nn.Sequential(convblock(in_channel = 8, channel_size = 32), 
                                    convblock(in_channel = 32, channel_size = 64),
                                    convblock(in_channel = 64, channel_size = 64),
                                    convblock(in_channel = 64, channel_size = 32), 
                                    convblock2(in_channel = 32, channel_size = self.c), 
                                   )

        self.decode_1d = conv1dblock(in_channel=8, channel_size = 64)
        

        #run forward the network to get to know the first value in the list passed to MLP (here 344)
        #self.ffblock = MLP([64, 128, 464])
        #self.ffblock = MLP([n_hidden, self.mlps3, self.mlps2, self.mlps1])
        self.ffblock = MLP([n_hidden, self.mlps1])

    def forward(self, x, y = None):
        '''
        x = self.ffblock(x)
        x = x.view(-1,8,58)
        x = self.decode_1d(x)
        #print("decode forward shape ",x.shape)
        x = x.view(-1,64,8,8)
        x = self.decode(x)
        
        
        return x
        '''


        x = self.ffblock(x)
        x = x.view(-1,self.ec,self.eh, self.ew)
        #print("output after ff block ",x.shape)
        x = self.decode(x)
        
        
        return x


if __name__ == '__main__':
    bsize = 4
    imshape = (32,32,1)

    cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    d = Decoder(shape = imshape).to(device)


    print(sum(p.numel() for p in d.parameters())/1e6, 'M parameters')
    t = torch.randn((bsize, 64)).to(device)

    out = d(t)
    print("model out shape ",out.shape)