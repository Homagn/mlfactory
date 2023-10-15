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


from models.pytorch.gpt import BigramLanguageModel
from models.pytorch import bvae
from dataloaders.imgjsonloader import jsonlabel_loader



def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img



#this function will change based on type of application the dataloader is being used for
#specifies how to load and process image and labels from specified json dictionary element
#function is required to initialize jsonlabel_loader
def process_dict_ele(folder, elem):
    #fname = '/datasets/behavior_cloning/game1/'+elem["id"]
    fname = folder+elem["id"]

    x = cv2.imread(fname, 0) #read as grayscale
    x = center_crop(x,(128,128))
    x = cv2.resize(x,(64,64))

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

    x = np.array(x/255.0, dtype = np.float32).reshape((1,64,64))

    return x, y

@torch.no_grad()
def get_latents(encoder, full_state, sequence_len, latent_size):
    store = torch.zeros((full_state.shape[0],full_state.shape[1],latent_size)).to("cuda")
    #bias = 2.0*torch.ones((full_state.shape[0],full_state.shape[1],latent_size)).to("cuda")
    for i in range(sequence_len):
        #encoded_mean_x, encoded_var_x = encoder.encoder(full_state[:,i,:,:,:])
        
        distributions = encoder._encode(full_state[:,i,:,:,:])
        mu = distributions[:, :encoder.z_dim]
        logvar = distributions[:, encoder.z_dim:]
        z = bvae.reparametrize(mu, logvar)

        #store[:,i,:] = state_encoder.reparameterization(encoded_mean_x, torch.exp(0.5 * encoded_var_x)) # takes exponential function (log var -> var)
        store[:,i,:] = z
    #return (1/3.0)*(store+bias) #try to make sure in range 0 to 1 for proper input and not -1 to 1
    return store #try to make sure in range 0 to 1 for proper input and not -1 to 1


if __name__ == '__main__':
    
    datafolders = [1,2,3,4,5,6,7,8, 9, 10, 11, 12, 13, 14, 15] #each number stores a single run of a game

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    context_len = 60
    C,D = 1, 64 #channels and full state or image size
    lr = 0.0005
    latent_size = 10

    jloaders = {}
    for j in datafolders:
        jl = jsonlabel_loader('/datasets/behavior_cloning/maze_game/game'+str(j)+'/','samplelabels.json', process_dict_ele)
        jl.discard_ini_sequence_len = 300 #85, 0, 200, 40
        jl.discard_final_sequence_len = 300 # 178, 131, 63, 92
        jl.skip_sampling = 2 #consequtive frames in sampled sequence are actually 10 frames apart in the raw collected data
        jl.uniform_sampled_label = [] #use this to focus sampling sequences whose end actions are the required action numbers in the list, if nothing then uniform
        jloaders[j] = jl

    model = BigramLanguageModel(n_embd = latent_size, block_size = context_len, action_size = 9, n_heads = 10, depth = 8, dropout = 0.1, device = device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    model.to(device)

    if os.path.exists('gpt.pt'):
        print("loading weights do far ")
        model = torch.load('gpt.pt')
    model.to(device)
    

    # Training loop
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()


    state_encoder = torch.load('../variational_encoders/results/vae_weights66.pt').to(device)
    state_encoder.eval() #make sure dropout and batchnorms are not being used 
    print("loaded state encoder model ")
    

    for n in range(10000):
        loader = np.random.choice(datafolders)
        xb, yb = jloaders[loader].sample_batch(bsize = batch_size, sequence_len = context_len+1, print_sample = False)
        
        x = torch.from_numpy(xb[:,:context_len,:,:,:]).view((batch_size, context_len, C, D, D)).to(device)
        xl = get_latents(state_encoder, x, context_len, latent_size).to(device)

        #print("got latents ", xl.shape)
        #print("got y shape ",yb.shape)
        y = torch.tensor(yb[:,1:context_len+1], dtype=torch.long).to(device)

        logits, loss = model(xl, y)
        #optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print("got logits ",logits.shape)
        print("got loss ",loss.item())

        if n%10==0:
            print("saving latent gpt model ")
            torch.save(model, 'gpt.pt')





    


    