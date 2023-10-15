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

import threading

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
from models.pytorch.gpt import visgpt
from models.pytorch import bvae
from datetime import datetime as dt
import cv2
from collections import deque

import mss.tools #for fast screenshots
import pyautogui
from datetime import datetime as dt
import time
import mss.tools #pip install mss

pressed = []
pred = 8
agent_alive = True

def agent_action(): #maybe need to incorporate multinomial sampling
    global pressed
    global pred
    global agent_alive

    while agent_alive:
        #pred = int(np.load("pred.npy"))
        #print("loaded pred ", pred)
        if pressed!=[]:
            for p in pressed:
                pyautogui.keyUp(p)

        if pred==0:
            pyautogui.keyDown('w') # for example key = 'a'
            #pyautogui.press('w') # for example key = 'a'
            pressed = ['w']
        if pred==1:
            pyautogui.keyDown('a') # for example key = 'a'
            #pyautogui.press('a') # for example key = 'a'
            pressed = ['a']
        if pred==2:
            pyautogui.keyDown('s') # for example key = 'a'
            #pyautogui.press('s') # for example key = 'a'
            pressed = ['s']
        if pred==3:
            pyautogui.keyDown('d') # for example key = 'a'
            #pyautogui.press('d') # for example key = 'a'
            pressed = ['d']


        if pred==4:
            pyautogui.keyDown('w') # for example key = 'a'
            pyautogui.keyDown('a') # for example key = 'a'
            #pyautogui.press('w') # for example key = 'a'
            #pyautogui.press('a') # for example key = 'a'
            pressed = ['w','a']
        if pred==5:
            pyautogui.keyDown('w') # for example key = 'a'
            pyautogui.keyDown('d') # for example key = 'a'
            #pyautogui.press('w') # for example key = 'a'
            #pyautogui.press('d') # for example key = 'a'
            pressed = ['w','d']
        if pred==6:
            pyautogui.keyDown('s') # for example key = 'a'
            pyautogui.keyDown('a') # for example key = 'a'
            #pyautogui.press('s') # for example key = 'a'
            #pyautogui.press('a') # for example key = 'a'
            pressed = ['s','a']
        if pred==7:
            pyautogui.keyDown('s') # for example key = 'a'
            pyautogui.keyDown('d') # for example key = 'a'
            #pyautogui.press('s') # for example key = 'a'
            #pyautogui.press('d') # for example key = 'a'

            pressed = ['s','a']

def close_agent():
    global pressed

    if pressed!=[]:
        for p in pressed:
            pyautogui.keyUp(p)




def record_screen(rs):
    #im = pyautogui.screenshot(region=(823, 153, 997, 464)) #takes around 0.15 s
    #cv2.imwrite(folder+"/"+str(idx)+".png", np.array(im))
    #print("time before mss ",dt.now())
    with mss.mss() as sct:
        # The screen part to capture
        monitor = {"top": 191, "left": 373, "width": 1125, "height": 611}
        #output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)
        output = "temp.png"

        '''
        # Grab the data
        sct_img = sct.grab(monitor) #takes around 0.03 s
        # Save to the picture file
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        '''

        # Grab the data
        sct_img = np.array(sct.grab(monitor)) #takes around 0.03 s

        #cv_img = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR)
        cv_img = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2GRAY)
        cv_img = cv2.resize(cv_img, (rs,rs))

        


        #print(output)
    #print("time after mss ",dt.now())

    #belo two operations take 0.003 seconds
    #im = cv2.imread("temp.png", 0)
    #im = cv2.resize(im, (rs,rs))

    #return im

    #verified imge recording to be ok

    #cv2.imshow("recorded image ",cv_img)
    #cv2.waitKey(1)
    
    return cv_img




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
    print("loading visgpt ")
    C,D = 1, 256 #input image properties, assuming H=W=D
    batch_size = 1
    n_class = 9
    latent_size = 256
    context_len = 30 #history accesible to the agent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = visgpt(n_embd = latent_size, block_size = context_len, action_size = 9, n_heads = 8, depth = 8, dropout = 0.1, device = device)

    if os.path.exists('gpt.pt'):
        print("loading weights do far ")
        model = torch.load('gpt.pt')

    
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    model.to(device)
    model.eval() #make sure the dropouts are not being used

    state_encoder = torch.load('../variational_encoders/results/vae_weights58.pt').to(device)
    state_encoder.eval() #make sure the batchnorm and dropouts are not being used
    print("loaded state encoder model ")


    #start a thread to that keeps executing keyboard actions in background
    thread = threading.Thread(target=agent_action, args=())
    thread.start()

    
    time.sleep(5) #provide time for user to position the mouse cursor
    y = 8

    gpt_input = None

    for _ in range(200): #try out 100 model actions
        #print("record screen start ",dt.now())

        x = record_screen(D) #roughly 0.02 seconds
        #print("record screen end ",dt.now())
        
        
        #below block takes 0.001 s roughly
        x = np.array(x/255.0, dtype = np.float32).reshape((C,D,D))
        x = torch.from_numpy(x).view((1, 1, C, D, D)).to(device) #batch size =1 , time frames =1 (do it time step wise)
        xl = x #get_latents(state_encoder, x, 1, latent_size).to(device) # ( batch_size, time, encoded latent size )
        

        
        #below block takes around 0.0002 seconds
        if gpt_input==None:
            gpt_input = xl
        else:
            gpt_input = torch.cat((gpt_input, xl), dim=1) # ( batch_size, time+1, encoded latent size )
            if gpt_input.shape[1]>context_len:
                gpt_input = gpt_input[:, -context_len:, :]

        
        
        
        out,_ = model(gpt_input) #roughly 0.01 seconds 
        
        
        
        #print("got out ",out)
        # focus only on the last time step
        out = out[:, -1, :] # becomes (B, C)
        probs = F.softmax(out, dim=-1) 
        #print("got probs ",probs)
        pred = torch.multinomial(probs, num_samples=1).detach().cpu().numpy()[0][0] # (B, 1)

        


        
        


        print("agent prediction index and time instant ",pred, dt.now())
        time.sleep(0.02)
        #np.save("pred.npy",pred)

        
        #agent_action(pred) #because of pyautogui this function takes 0.2 seconds so new to run this as a seperate thread
        
        
    pred = 8
    time.sleep(1)
    close_agent()
    agent_alive = False
    thread.join()
    close_agent()



