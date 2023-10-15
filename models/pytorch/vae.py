import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.optim import Adam


import os,sys
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


from models.pytorch.conv_reducer import Encoder as Encoder_image
from models.pytorch.conv_upsampler import Decoder as Decoder_image



"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var






class Encoder_2d(nn.Module):
    
    def __init__(self, hidden_dim, latent_dim, imshape = (128,128,3)):
        super(Encoder_2d, self).__init__()

        self.enc2d = Encoder_image(shape = imshape)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.sigm = nn.Sigmoid() #constrain the output of encoder to be in the 0 to 1 range
        self.relu = nn.ReLU()
        
        self.training = True
        
    def forward(self, x):
        h = self.enc2d(x)
        h = self.LeakyReLU(self.FC_input2(h))
        #h = self.relu(self.FC_input2(h))
        #h = self.sigm(self.FC_input2(h))
        mean     = self.FC_mean(h)
        log_var  = self.FC_var(h) 

        return mean, log_var


class Decoder_2d(nn.Module):
    def __init__(self, latent_dim, hidden_dim, imshape = (128,128,3)):
        super(Decoder_2d, self).__init__()

        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.final_proj = nn.Conv2d(imshape[2], imshape[2], 1, padding = 0) #similar to applying linear after fully connected
        self.final_act = nn.Sigmoid()
        self.dec2d = Decoder_image(shape = imshape)

        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x): #do not use leaky relu because its a weak non linearity, useful only when too much noise in data
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))

        #h     = self.relu(self.FC_hidden(x))
        #h     = self.relu(self.FC_hidden2(h))
        
        x_hat = self.final_proj(self.dec2d(h))
        x_hat = self.final_act(x_hat)
        
        return x_hat

class vae_conv(nn.Module):
    def __init__(self, Encoder, Decoder, device = "cuda"):
        super(vae_conv, self).__init__()
        self.Encoder = Encoder #Encoder_2d
        self.Decoder = Decoder #Decoder_2d
        self.device = device
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var








def evaluate_model(model,test_loader, DEVICE, batch_size, x_dim):
    import matplotlib
    #apt-get install python3-tk
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    model.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            
            x_hat, _, _ = model(x)


            break

    def show_image(x, idx):
        x = x.view(batch_size, 28, 28)

        fig = plt.figure()
        plt.imshow(x[idx].cpu().numpy())
        print("showing model inference on training set")
        plt.show()

    show_image(x, idx=0)
    show_image(x_hat, idx=0)


def test_generation(batch_size, latent_dim, decoder, DEVICE):
    print("generating image from noise")
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)

    save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')
    print("saved as generated_sample.png")



if __name__ == '__main__':
    


    # Model Hyperparameters
    dataset_path = '~/datasets'
    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")

    batch_size = 100
    x_dim  = 784
    hidden_dim = 400
    latent_dim = 200
    lr = 1e-3
    epochs = 30




    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader


    mnist_transform = transforms.Compose([
            transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} 

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)






    #setup model
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)





    #setup loss
    BCE_loss = nn.BCELoss()
    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD
    optimizer = Adam(model.parameters(), lr=lr)







    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        
    print("Finish!!")




    #test performance
    evaluate_model(model,test_loader, DEVICE, batch_size, x_dim)
    test_generation(batch_size, latent_dim, decoder, DEVICE)








