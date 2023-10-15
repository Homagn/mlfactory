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
#from dataloaders.imgjsonloader import jsonlabel_loader
from torch.optim import Adam
import cv2
import numpy as np
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from tqdm import tqdm




def test_generation(batch_size, latent_dim, model, DEVICE, epoch):
    print("generating image from noise")
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = model.decoder(noise)

    save_image(generated_images.view(batch_size, 1, 64, 64), 'results/generated_sample'+str(epoch)+'.png')
    print("saved as generated_sample.png")


def test_reconstruction(x, model, epoch):
    print("reconstructing input")
    with torch.no_grad():
        x_hat, mean, log_var = model(x)
    print("x_hat shape ",x_hat.shape)
    save_image(x_hat.view(batch_size, 1, 64, 64), 'results/reconstructed_sample'+str(epoch)+'.png')
    print("saved as reconstructed_sample.png")






if __name__ == '__main__':

    num_batches = 1000000
    #batch size is very important in vae if reconstruction is uniform color and lacks any detail reduce the batch size
    #for maze game use batch size = 2, for car racing game (less diverse images) use 16
    #training may be required to do multiple times from various starting points, because sometimes it just converge to uniform gray image, so just start training again
    batch_size = 2 
    lr = 0.0003
    datafolders = [1,2,3,4,5,6,7,8, 9, 10, 11, 12, 13, 14, 15] #each number stores a single run of a game
    image_generation_shape = (64,64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")








    #first test with stl 10
    import torchvision
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    import sys
    import cv2


    
    kwargs = {'num_workers': 4, 'pin_memory': True} 
    data_transform = transforms.Compose([
                transforms.CenterCrop( (128,128)),
                transforms.Resize(size = (64,64)),
                transforms.Grayscale(),
                transforms.ToTensor(),
        ])

    #cifar_trainset = datasets.ImageFolder(root='/datasets/behavior_cloning/sampled', transform=cifar_transform)

    data_trainset = datasets.ImageFolder(root='/datasets/behavior_cloning/maze_game', transform=data_transform)

    train_loader = DataLoader(dataset=data_trainset, batch_size=batch_size, shuffle=True, **kwargs) 

    






    model = bvae.BetaVAE_H(nc=1).to(device) # by default the model is made to only accept 64x64 size image input
    optimizer = Adam(model.parameters(), lr=lr)

    
    
    print("Start training VAE on my dataset ...")
    model.train()

    x_last = None

    for e in range(num_batches):
        overall_loss = 0 
        for batch_idx, (x, _) in enumerate(tqdm(train_loader)):
            #x = x.permute(0,2,3,1)
            #print("x shape ",x.shape)
            try:
                x = x.view((batch_size,1,64,64)).to(device, non_blocking = True)
                #x = x.to(device)
                optimizer.zero_grad() #zero grad the generator optimizer
                x_recon, mu, logvar = model(x)
                recon_loss = bvae.reconstruction_loss(x, x_recon, 'gaussian')
                total_kld, dim_wise_kld, mean_kld = bvae.kl_divergence(mu, logvar)

                beta_vae_loss = recon_loss + 4.0*total_kld # hardcode the beta term to default =4

                beta_vae_loss.backward()

                optimizer.step()

                overall_loss += beta_vae_loss.item()
                x_last = x

                #print("got beta vae loss ",beta_vae_loss)
            
            except:
                print("batch corrupted passing")
        print("\tEpoch", e + 1, "complete!", "\tTotal Loss: ", overall_loss )
        
        print("testing generation ")
        test_generation(batch_size, 10, model, device, e) #latent size = 10 by default

        print("testing reconstruction ")
        test_reconstruction(x_last,model, e)

        print("saving model ")
        torch.save(model, 'results/vae_weights'+str(e)+'.pt')
    
        
    print("Finish!!")
    