from pathlib import Path

import torch
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


from models.pytorch import vae
from dataloaders.imgjsonloader import jsonlabel_loader
from torch.optim import Adam
import cv2
import numpy as np
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

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

    #stack the current action to current state image as an image
    
    #r = np.random.randint(0,9) #consider 20 random generating modes
    #action_conditioning = (r/9.0)*np.ones((112,112), dtype = np.float32).reshape((1,112,112))

    #action_conditioning = np.random.sample((1,112,112)).astype(np.float32) #apply a random conditioning to each frame to help the network be able to think generatively


    x = np.array(x/255.0, dtype = np.float32).reshape((1,256,256))

    return x, y


def test_generation(batch_size, latent_dim, decoder, DEVICE):
    print("generating image from noise")
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)

    save_image(generated_images.view(batch_size, 1, 256, 256), 'generated_sample.png')
    print("saved as generated_sample.png")


def test_reconstruction(x, model):
    print("reconstructing input")
    with torch.no_grad():
        x_hat, mean, log_var = model(x)
    print("x_hat shape ",x_hat.shape)
    save_image(x_hat.view(batch_size, 1, 256, 256), 'reconstructed_sample.png')
    print("saved as reconstructed_sample.png")


if __name__ == '__main__':

    num_batches = 1000000
    batch_size = 16
    lr = 0.0003
    datafolders = [1,2,3,4,5,6,7,8, 9, 10, 11] #each number stores a single run of a game
    image_generation_shape = (256,256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jloaders = {}
    for j in datafolders:
        jl = jsonlabel_loader('/datasets/behavior_cloning/game'+str(j)+'/','samplelabels.json', process_dict_ele)
        jl.discard_ini_sequence_len = 50 #85, 0, 200, 40
        jl.discard_final_sequence_len = 50 # 178, 131, 63, 92
        jl.uniform_sampled_label = [1,2,3]
        jloaders[j] = jl








    #first test with stl 10
    import torchvision
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    import sys
    import cv2


    
    kwargs = {'num_workers': 4, 'pin_memory': True} 
    cifar_transform = transforms.Compose([
                transforms.Resize(size = (256,256)),
                transforms.Grayscale(),
                transforms.ToTensor(),
        ])

    cifar_trainset = datasets.ImageFolder(root='/datasets/behavior_cloning/sampled', transform=cifar_transform)

    train_loader = DataLoader(dataset=cifar_trainset, batch_size=batch_size, shuffle=True, **kwargs) 

    












    

    #setup model
    encoder = vae.Encoder_2d( hidden_dim=64, latent_dim=32, imshape = (256,256,1))
    decoder = vae.Decoder_2d(latent_dim=32, hidden_dim = 64, imshape = (256,256,1))
    model = vae.vae_conv(Encoder=encoder, Decoder=decoder, device = device).to(device)

    if os.path.exists('enc_weights.pt'):
        print("loading encoder weights do far ")
        encoder = torch.load('enc_weights.pt')

    if os.path.exists('dec_weights.pt'):
        print("loading decoder weights do far ")
        decoder = torch.load('dec_weights.pt')

    if os.path.exists('vae_weights.pt'):
        print("loading vae weights do far ")
        model = torch.load('vae_weights.pt')

    model = model.to(device)


    
    
    

    '''
    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.MSELoss()(x_hat, x)
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD
    '''
    
    #pip install pytorch-msssim
    from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim


    class SSIM_Loss(SSIM):
        def forward(self, img1, img2):
            return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )

    #loss_function = SSIM_Loss(data_range=1.0, size_average=True, channel=3)

    #use BCE when output last layer has sigmoid otherwise youll get complex cuda errors
    #setup loss
    def loss_function(x, x_hat, mean, log_var):
        #reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
        reproduction_loss = nn.BCELoss(size_average=False)(x_hat, x) / x.size(0)
        KLD      = - 0.5 * torch.mean(1+ log_var - mean.pow(2) - log_var.exp())
        #ssim = SSIM_Loss(data_range=1.0, size_average=True, channel=1)(x_hat,x)
        #mse = nn.MSELoss()(x_hat, x)
        return reproduction_loss + KLD #+ ssim +mse


    optimizer = Adam(model.parameters(), lr=lr)

    
    '''
    print("Start training VAE...")
    model.train()

    for n in range(num_batches):
        loader = np.random.choice(datafolders)
        #sample a random batch (iid) out of any of the game folders
        xb, yb = jloaders[loader].sample_batch(bsize = batch_size, sequence_len = 1, print_sample = False)
        
        optimizer.zero_grad() #zero grad the generator optimizer

        x = torch.from_numpy(xb).view((batch_size, 1, 256, 256)).to(device)

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        #loss = loss_function(x, x_hat)
        
        #print("got x_hat ",x_hat.shape)
        
        loss.backward()
        optimizer.step()

        print("loss ",loss.item())

        if n%600==0:
            print("testing generation for epoch ",n//600)
            test_generation(batch_size, 32, decoder, device)

            print("testing reconstruction ")
            test_reconstruction(x,model)

            print("saving model ")
            torch.save(model, 'vae_weights.pt')
            torch.save(encoder, 'enc_weights.pt')
            torch.save(decoder, 'dec_weights.pt')
    '''








    
    #first finetune on stl10 dataset
    
    #print("Start training VAE on stl10 ...")
    print("Start training VAE on my dataset ...")
    model.train()
    x_last = None
    for e in range(num_batches):
        overall_loss = 0 
        for batch_idx, (x, _) in enumerate(tqdm(train_loader)):
            #x = x.permute(0,2,3,1)
            #print("x shape ",x.shape)
            try:
                x = x.view((batch_size,1,256,256)).to(device, non_blocking = True)
                #x = x.to(device)
                optimizer.zero_grad() #zero grad the generator optimizer
                x_hat, mean, log_var = model(x)
                loss = loss_function(x, x_hat, mean, log_var)
                #loss = loss_function(x, x_hat)
                
                #print("got x_hat ",x_hat.shape)
                
                loss.backward()
                optimizer.step()

                overall_loss += loss.item()
                x_last = x
            except:
                print("batch corrupted passing")
        print("\tEpoch", e + 1, "complete!", "\tTotal Loss: ", overall_loss )
        print("testing generation ")
        test_generation(batch_size, 32, decoder, device)

        print("testing reconstruction ")
        test_reconstruction(x_last,model)

        print("saving model ")
        torch.save(model, 'vae_weights.pt')
        torch.save(encoder, 'enc_weights.pt')
        torch.save(decoder, 'dec_weights.pt')
    
        
    print("Finish!!")
    