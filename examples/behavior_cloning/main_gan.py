from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import os, sys
import cv2
import numpy as np

# An installation agnostic method to find and link to root of the package which is mlfactory
#==========================================================
import re
try: #testing the functions locally without pip install
  import __init__
  cimportpath = os.path.abspath(__init__.__file__)
  if 'extensions' in cimportpath:
    print("local testing ")
    import mlfactory
    cimportpath = os.path.abspath(mlfactory.__file__)

except: #testing while mlfactory is installed using pip
  print("Non local testing")
  import mlfactory
  cimportpath = os.path.abspath(mlfactory.__file__)

main_package_loc = cimportpath[:cimportpath.rfind('mlfactory')+len('mlfactory')]
print("got main package location ",main_package_loc)


os.environ['top'] = main_package_loc
sys.path.append(os.path.join(os.environ['top']))
#==========================================================


from models.pytorch.timesformer import TimeSformer
from dataloaders.imgjsonloader import jsonlabel_loader

#pip install focal_loss_torch
from focal_loss.focal_loss import FocalLoss


#this function will change based on type of application the dataloader is being used for
#specifies how to load and process image and labels from specified json dictionary element
#function is required to initialize jsonlabel_loader
def process_dict_ele(folder, elem):
    #fname = '/datasets/behavior_cloning/game1/'+elem["id"]
    fname = folder+elem["id"]

    x = cv2.imread(fname, 0) #read as grayscale
    x = cv2.resize(x,(112,112))

    k = elem['keys_pressed']
    y = 0 #8 cardinal directions movement and do nothing

    if k == ["'w'"]:
        y = 0 
    if k == ["'a'"]:
        y = 1
    if k == ["'s'"]:
        y = 2
    if k == ["'d'"]:
        y = 3

    if "'w'" in k and "'a'" in k:
        y = 4
    if "'w'" in k and "'d'" in k:
        y = 5
    if "'s'" in k and "'a'" in k:
        y = 6
    if "'s'" in k and "'d'" in k:
        y = 7

    if k == []:
        y = 8

    #stack the current action to current state image as an image
    
    #r = np.random.randint(0,9) #consider 20 random generating modes
    #action_conditioning = (r/9.0)*np.ones((112,112), dtype = np.float32).reshape((1,112,112))

    #action_conditioning = np.random.sample((1,112,112)).astype(np.float32) #apply a random conditioning to each frame to help the network be able to think generatively


    x = np.array(x/255.0, dtype = np.float32).reshape((1,112,112))
    #x = np.stack((x,action_conditioning)).reshape((2,112,112))
    x = np.stack((x,x)).reshape((2,112,112))

    #y = np.array(y, dtype = np.int32)
    #print("output of data processing func y ",y)

    return x, y



if __name__ == '__main__':
    print("loading timesformer ")
    num_prev_conditioning_frames = 2

    train_traj_len = 7 #loss would be calculated as a multiclass sequence loss so agent is not penelized over a single action, but a trajectory of action

    C,D = 2, 112 #input image properties, assuming H=W=D
    batch_size = 2
    n_class = 9
    num_batches = 2500
    LR = 0.0003
    datafolders = [1,2,3,4,5,6,7,8] #each number stores a single run of a game

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jloaders = []
    for j in datafolders:
        jl = jsonlabel_loader('/datasets/behavior_cloning/game'+str(j)+'/','samplelabels.json', process_dict_ele)
        jl.discard_ini_sequence_len = 20 #85, 0, 200, 40
        jl.discard_final_sequence_len = 100 # 178, 131, 63, 92
        jloaders.append(jl)

    model = TimeSformer(
    dim = 512,
    image_size = D,
    patch_size = 8,
    channels = C,
    num_frames = num_prev_conditioning_frames,
    num_classes = n_class,
    depth = 8,
    heads = 8,
    dim_head =  64,
    attn_dropout = 0.1,
    ff_dropout = 0.1
    )

    model_disc = TimeSformer(
    dim = 256,
    image_size = D,
    patch_size = 8,
    channels = C,
    num_frames = train_traj_len,
    num_classes = 1,
    depth = 8,
    heads = 8,
    dim_head =  64,
    attn_dropout = 0.1,
    ff_dropout = 0.1
    )

    if os.path.exists('weights.pt'):
        print("loading weights do far ")
        model = torch.load('weights.pt')

    
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print(sum(p.numel() for p in model_disc.parameters())/1e6, 'M parameters')

    

    model.to(device)
    model_disc.to(device)
    

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    optimizer_disc = Adam(model_disc.parameters(), lr=LR)
    
    #criterion = CrossEntropyLoss()
    #criterion = FocalLoss(gamma=0.7)
    criterion_GD = nn.BCELoss()
    

    train_loss = 0.0
    for n in range(num_batches):
        #pick a random game run out of all the available runs
        loader = np.random.choice(datafolders)

        #sample a random batch sequence out of any of the game folders
        xb, yb = jloaders[loader-1].sample_batch(bsize = batch_size, sequence_len = (num_prev_conditioning_frames+train_traj_len), print_sample = False)
        

        y_hat = None
        y_targ = None

        optimizer.zero_grad() #zero grad the generator optimizer

        #go forward in chunks of num_prev_conditioning_frames
        #feed those chunks to the generator and get the agent predicted action
        #do this for train_traj_len to generate an agent action trajectory stored in y_hat
        #y_targ stores the true tracjectory
        for t in range(train_traj_len):
            xbt = xb[:, t:t+num_prev_conditioning_frames, :, :, :]
            ybt = yb[:,num_prev_conditioning_frames+t].reshape(batch_size,1)

            #print("xb shape ",xbt.shape)

            
            x = torch.from_numpy(xbt).view((batch_size, num_prev_conditioning_frames, C, D, D)).to(device)
            y = torch.tensor(ybt, dtype=torch.long).to(device)

            if y_targ==None:
                y_targ = y
            else:
                y_targ = torch.cat((y_targ,y), dim=1)

            yp = model(x)

            if y_hat==None:
                y_hat = yp
            else:
                y_hat = torch.cat((y_hat, yp)) 


        y_hat = y_hat.view(batch_size, train_traj_len, n_class)
        y_targ = y_targ.view(batch_size, train_traj_len)

        xbt_d = xb[:, num_prev_conditioning_frames:(train_traj_len+num_prev_conditioning_frames), :, :, :]
        xbt_d = torch.from_numpy(xbt_d).view((batch_size, train_traj_len, C, D, D)).to(device)

        #generator generates action sequence trajectory which is now used to condition the inputs received by the discriminator
        y_sampled = torch.argmax(F.softmax(y_hat, dim=-1), dim=-1)/n_class
        y_sampled = y_sampled.view(batch_size,train_traj_len, 1,1,1)
        
        #applying generator conditioning to discriminator inputs
        action_cond = y_sampled*torch.ones((batch_size, train_traj_len, 1,112,112)).to(device)
        xbt_d[:,:,1:,:,:] = action_cond
        
        

        ybt_d = nn.Sigmoid()(model_disc(xbt_d)).squeeze(1)

        disc_targ = torch.ones(batch_size).to(device)
        #print("model disc out shape ",ybt_d.shape)
        loss_g = criterion_GD(ybt_d, disc_targ)
        #print("Generator + Discriminator model output ",ybt_d)
        
        print("Generator loss ",loss_g.detach().cpu().item() )

        
        loss_g.backward()
        optimizer.step()


        optimizer_disc.zero_grad() #zero grad the generator optimizer
        #preparing inputs to train the discriminator

        xbt_t = xb[:, num_prev_conditioning_frames:(train_traj_len+num_prev_conditioning_frames), :, :, :]
        xbt_t = torch.from_numpy(xbt_t).view((batch_size, train_traj_len, C, D, D)).to(device)
        
        #applying true user demonstration conditioning on the trajectory
        y_targ = y_targ.view(batch_size,train_traj_len, 1,1,1)
        action_cond_t = y_targ*torch.ones((batch_size, train_traj_len, 1,112,112)).to(device)
        xbt_t[:,:,1:,:,:] = action_cond
        
        
        true_disc_out = nn.Sigmoid()(model_disc(xbt_t)).squeeze(1)
        true_loss_d = criterion_GD(true_disc_out ,disc_targ)
        print("true discriminator loss ",true_loss_d.detach().cpu().item())

        

        gd_out = nn.Sigmoid()(model_disc(xbt_d.detach())).squeeze(1)
        #gd_out = nn.Sigmoid()(gd_out)
        gd_loss = criterion_GD(gd_out, torch.zeros(batch_size).to(device))
        print("generator discriminator loss ",gd_loss.detach().cpu().item())
        total_disc_loss = (gd_loss + true_loss_d) /2
        total_disc_loss.backward()
        optimizer_disc.step()

        print("Batch number ",n)

        if n%10==0:
            print("saving model ")
            torch.save(model, 'weights.pt')


