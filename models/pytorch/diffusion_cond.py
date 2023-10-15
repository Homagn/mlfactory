import sys, os
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


import torch
import torchvision
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import sys,os
from PIL import Image

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np


BATCH_SIZE = 2
IMG_SIZE = 64
img_channels = 3

#===================================================================================
#functions to show PIL images and pytorch tensor images

def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, (img,_) in enumerate(data):
        if i == num_samples:
            break
        img = img.permute(0,2,3,1)
        img = img.cpu().numpy()

        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0].reshape((64,64,3)))
        plt.show()


def show_tensor_image(image, show = False):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    if show:
        plt.imshow(reverse_transforms(image))
        plt.show()
    else:
        plt.imshow(reverse_transforms(image))
    return plt



#===================================================================================




# Noise scheduler
#=================================================================
import torch.nn.functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

#=================================================================




#===================================================================================
#define dataloader
from dataloaders import trimesh_render


#===================================================================================


#===================================================================================
# Simulate forward diffusion (uncomment to visualize the forward process)
# checked to work fine
'''
print("simulation diffusion forward process ")
import cv2

image_r, image_c, label = trimesh_render.sample_batch(sz=1)
image = image_r #get rid of the batch dimension
#print("image_r shape ",image_r.shape, image.shape)



plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)

    img, noise = forward_diffusion_sample(image, t)
    show_tensor_image(img, show = True)

sys.exit(0)
'''

#===================================================================================







#===================================================================================

#define the model
from models.pytorch import xunet_cond


model = xunet_cond.UNet_conditional_labelimage(c_in = img_channels, c_out = img_channels, num_classes= 152)
print("Num params of the model : ", sum(p.numel() for p in model.parameters()))
#can check the model architecture as follows
#print(model)
#===================================================================================





#===================================================================================
# define loss 

def get_loss(model, x, xc, t, labels):
    x_noisy, noise = forward_diffusion_sample(x, t, device)
    noise_pred = model(x, xc, t, labels)
    return F.l1_loss(noise, noise_pred)
#===================================================================================





#===================================================================================
#sampling 

@torch.no_grad()
def sample_timestep(x, xc, t, label):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, xc, t, label) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

            

@torch.no_grad()
def sample_plot_saveimage(image_number = 0):
    # Sample noise
    import matplotlib.pyplot as plt
    img_size = IMG_SIZE
    img = torch.randn((1, img_channels, img_size, img_size), device=device)

    label = torch.Tensor([np.random.randint(152)] * 1).long().to(device)
    _,img_c,_ = trimesh_render.sample_batch(sz=1)
    
    #img_c = np.asarray(img_c).astype(np.float32)
    #img_c = torch.from_numpy(img_c).view((1, 1, img_size, img_size)).to(device)
    img_c = img_c.to(device)


    plt.figure(figsize=(15,15))
    plt.axis('off') 
    num_images = 5
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        print("generating ",i)
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, img_c, t, label)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            plt = show_tensor_image(img.detach().cpu(), show = False)
    #plt.show() 
    plt.savefig('results/model_output'+str(image_number)+'.png')
    print("saved image")
#===================================================================================





#===================================================================================
# training loop

from torch.optim import Adam
from tqdm import tqdm
import sys,os

device = "cuda" if torch.cuda.is_available() else "cpu"


if os.path.exists('results/ddpm.pt'):
    model = torch.load('results/ddpm.pt')
    print("loaded previous model successfully ")
else:
    print("previous model does not exist")


model.to(device)
optimizer = Adam(model.parameters(), lr=0.0001)
epochs = 100000 # Try more!

for epoch in range(epochs):
    optimizer.zero_grad()
    xt, xr, l = trimesh_render.sample_batch(sz=BATCH_SIZE)

    xr = xr.to(device)
    xt = xt.to(device)
    l = l.to(device)
    t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
    loss = get_loss(model, xt, xr, t, l)

    loss.backward()
    optimizer.step()

    print("loss ",loss.item())

    if epoch % 500 == 0:
        

        
        sample_plot_saveimage(image_number = epoch)

        torch.save(model, 'results/ddpm.pt')
        print("saved model")






#===================================================================================