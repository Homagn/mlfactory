from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np
import cv2
import sys,os

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


from models.pytorch.gpt import visgpt
from models.pytorch import vae
from dataloaders.imgjsonloader import jsonlabel_loader



#this function will change based on type of application the dataloader is being used for
#specifies how to load and process image and labels from specified json dictionary element
#function is required to initialize jsonlabel_loader
def process_dict_ele(folder, elem):
    #fname = '/datasets/behavior_cloning/game1/'+elem["id"]
    fname = folder+elem["id"]

    x = cv2.imread(fname, 0) #read as grayscale
    x = cv2.resize(x,(256,256))

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

    x = np.array(x/255.0, dtype = np.float32).reshape((1,256,256))

    return x, y

if __name__ == '__main__':
    
    datafolders = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] #each number stores a single run of a game

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    context_len = 50
    C,D = 1, 256 #channels and full state or image size
    lr = 0.00005
    latent_size = 256

    jloaders = {}
    for j in datafolders:
        jl = jsonlabel_loader('/datasets/behavior_cloning/maze_game/game'+str(j)+'/','samplelabels.json', process_dict_ele)
        jl.discard_ini_sequence_len = 300 #85, 0, 200, 40
        jl.discard_final_sequence_len = 300 # 178, 131, 63, 92
        jl.skip_sampling = 2 #consequtive frames in sampled sequence are actually 10 frames apart in the raw collected data
        jl.uniform_sampled_label = [] #use this to focus sampling sequences whose end actions are the required action numbers in the list, if nothing then uniform
        jloaders[j] = jl

    model = visgpt(n_embd = latent_size, block_size = context_len, action_size = 9, n_heads = 8, depth = 8, dropout = 0.1, device = device)

    if os.path.exists('gpt.pt'):
        print("loading weights do far ")
        model = torch.load('gpt.pt')


    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    model.to(device)
    

    # Training loop
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()


    
    

    for n in range(20000):
        loader = np.random.choice(datafolders)
        xb, yb = jloaders[loader].sample_batch(bsize = batch_size, sequence_len = context_len+1, print_sample = False)
        
        x = torch.from_numpy(xb[:,:context_len,:,:,:]).view((batch_size, context_len, C, D, D)).to(device)
        
        y = torch.tensor(yb[:,1:context_len+1], dtype=torch.long).to(device) #y is the prediction of the future so its shifted by 1 step

        logits, loss = model(x, y)
        #optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print("got logits ",logits.shape)
        print("got loss ",loss.item())

        if n%10==0:
            print("saving latent gpt model for batch num ",n)
            torch.save(model, 'gpt.pt')





    


    