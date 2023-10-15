from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data
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


from models.pytorch import bvae
from dataloaders.imgjsonloader import jsonlabel_loader

from torch.optim import Adam
import cv2
import numpy as np
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from tqdm import tqdm



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







def show_reconstruction(x, model):
    print("reconstructing input")
    with torch.no_grad():
        #x_hat, mean, log_var = model(x[0].view((1,1,64,64)))
        x_hat, mean, log_var = model(x)
    print("x_hat shape ",x_hat.shape)
    xv = x_hat.cpu().numpy()
    print("xv shape ",xv.shape)
    print("showing entire sequence of length ",xv.shape[0])
    for i in range(xv.shape[0]):
        v = xv[i].reshape((64,64))

        a = x.cpu().numpy()
        actual = a[i].reshape((64,64))
        cv2.imshow("actual ",actual)
        cv2.imshow("reconstructed ",v)
        cv2.waitKey(0)
        print("mean ",mean[i])


def show_generation(batch_size, latent_dim, model, DEVICE):
    print("generating image from noise")
    decoder = model.decoder

    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)

    for b in range(batch_size):
        v = generated_images[b].cpu().numpy()
        v = v.reshape((64,64))
        cv2.imshow("generated ",v)
        cv2.waitKey(0)
    



if __name__ == '__main__':

    datafolders = [1,2,3,4,5,6,7,8, 9, 10, 11] #each number stores a single run of a game
    image_generation_shape = (256,256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #setup dataloader
    jloaders = {}
    for j in datafolders:
        jl = jsonlabel_loader('/datasets/behavior_cloning/maze_game/game'+str(j)+'/','samplelabels.json', process_dict_ele)
        jl.discard_ini_sequence_len = 500 #85, 0, 200, 40
        jl.discard_final_sequence_len = 500 # 178, 131, 63, 92
        jl.skip_sampling = 10 #consequtive frames in sampled sequence are actually 10 frames apart in the raw collected data
        jl.uniform_sampled_label = [] #use this to focus sampling sequences whose end actions are the required action numbers in the list, if nothing then uniform
        jloaders[j] = jl

    

    #setup model
    model = bvae.BetaVAE_H(nc=1).to(device)
    model = torch.load('results/vae_weights66.pt')

    model = model.to(device)

    loader = np.random.choice(datafolders)
    xb, yb = jloaders[loader].sample_batch(bsize = 4, sequence_len = 15, print_sample = False)
    print("shapes ",xb.shape, yb.shape)


    print("Start training VAE on my dataset ...")
    model.eval()


    inp = input("want to see generation or reconstruction ? (g/r) ")

    if inp=='r':
        for i in range(xb.shape[0]):
            print("xb shape ",xb.shape)
            xr = torch.from_numpy(xb[i]).view((15, 1,64,64)).to(device)
            show_reconstruction(xr,model)
    if inp=='g':
        print("now showing random generation ")
        show_generation(20, 10, model, device)