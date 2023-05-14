# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime as dt

#pip install torch-multi-head-attention
from torch_multi_head_attention import MultiHeadAttention

#simple regressor using 1d conv and fully connected layers
class reg1d(nn.Module):

    def __init__(self):
        super(reg1d, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.atten1 = MultiHeadAttention(in_features=200, head_num=20)
        self.atten2 = MultiHeadAttention(in_features=200, head_num=10)

        self.conv1 = nn.Conv1d(8, 32, 11, padding = "same")
        self.conv11 = nn.Conv1d(32, 32, 11, padding = "same")

        self.conv2 = nn.Conv1d(32, 32, 5, padding = "same")
        self.conv3 = nn.Conv1d(32, 64, 5, padding = "same")
        self.conv33 = nn.Conv1d(64, 64, 5, padding = "same")

        self.conv4 = nn.Conv1d(64, 128, 5, padding = "same")

        self.conv44 = nn.Conv1d(128, 128, 5, padding = "same")

        self.conv5 = nn.Conv1d(128, 64, 3, padding = "same")
        self.conv6 = nn.Conv1d(64, 32, 3, padding = "same")
        self.conv7 = nn.Conv1d(32, 1, 1, padding = "same")

        self.fc1 = nn.Linear(200,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 7)
        

    def forward(self, input):
        # Max pooling over a (2, 2) window
        #x = self.atten1(input,input,input, MultiHeadAttention.gen_history_mask(input))

        x = F.relu((self.conv1(input)))
        x = F.relu((self.conv11(x)))

        #print("x shape ",x.shape)

        x = self.atten1(x,x,x)
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv2(x)))
        
        x = self.atten1(x,x,x)

        x = F.relu((self.conv3(x)))
        x = F.relu((self.conv33(x)))

        x = self.atten1(x,x,x)



        x = F.relu((self.conv4(x)))
        x = F.relu((self.conv44(x)))
        x = F.relu((self.conv44(x)))
        x = F.relu((self.conv44(x)))

        x = self.atten1(x,x,x)


        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv6(x)))
        x = F.relu((self.conv7(x)))
        
        #print("x final shape ",x.shape)


        x = torch.nn.Flatten()(x)
        #print("x shape ",x.shape)
        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))
        x = F.relu((self.fc3(x)))

        
        return x


if __name__ == '__main__':
    net = reg1d()
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)


    input = torch.randn( 16, 8, 200) #batch size 16
    yt = torch.randn( 16, 7) #batch size 16

    print("Network start inference time ",dt.now())
    out = net(input)
    print("Network end inference time ",dt.now())
    print("out shape ",out.shape)

    optimizer.zero_grad()

    output = net(input)
    print("got model output ",output)
    print("got model target ",yt)
    loss = torch.nn.MSELoss()(output, yt)
    print("loss ",loss.data)
    #loss = torch.nn.MSELoss()(output, yt)

    #backward and step takes the most time around 1.2 sec
    loss.backward()
    optimizer.step()
    print("successfully made backward")