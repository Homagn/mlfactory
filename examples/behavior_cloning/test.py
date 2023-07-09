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
from datetime import datetime as dt
import cv2
from collections import deque

import mss.tools #for fast screenshots
import pyautogui
from datetime import datetime as dt
import time

pressed = []

def agent_action(pred): #maybe need to incorporate multinomial sampling
    global pressed

    if pressed!=[]:
        for p in pressed:
            pyautogui.keyUp(p)

    if pred==0:
        pyautogui.keyDown('w') # for example key = 'a'
        pressed = ['w']
    if pred==1:
        pyautogui.keyDown('a') # for example key = 'a'
        pressed = ['a']
    if pred==2:
        pyautogui.keyDown('s') # for example key = 'a'
        pressed = ['s']
    if pred==3:
        pyautogui.keyDown('d') # for example key = 'a'
        pressed = ['d']


    if pred==4:
        pyautogui.keyDown('w') # for example key = 'a'
        pyautogui.keyDown('a') # for example key = 'a'
        pressed = ['w','a']
    if pred==5:
        pyautogui.keyDown('w') # for example key = 'a'
        pyautogui.keyDown('d') # for example key = 'a'
        pressed = ['w','d']
    if pred==6:
        pyautogui.keyDown('s') # for example key = 'a'
        pyautogui.keyDown('a') # for example key = 'a'
        pressed = ['s','a']
    if pred==7:
        pyautogui.keyDown('s') # for example key = 'a'
        pyautogui.keyDown('d') # for example key = 'a'
        pressed = ['s','a']

def close_agent():
    global pressed

    if pressed!=[]:
        for p in pressed:
            pyautogui.keyUp(p)




def record_screen(rs):
    #im = pyautogui.screenshot(region=(823, 153, 997, 464)) #takes around 0.15 s
    #cv2.imwrite(folder+"/"+str(idx)+".png", np.array(im))

    with mss.mss() as sct:
        # The screen part to capture
        monitor = {"top": 191, "left": 373, "width": 1125, "height": 611}
        #output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)
        output = "temp.png"

        # Grab the data
        sct_img = sct.grab(monitor) #takes around 0.03 s

        # Save to the picture file
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        #print(output)
    im = cv2.imread("temp.png", 0)
    im = cv2.resize(im, (rs,rs))
    return im


if __name__ == '__main__':
    print("loading timesformer ")
    num_prev_conditioning_frames = 10
    C,D = 1, 112 #input image properties, assuming H=W=D
    batch_size = 1
    n_class = 9
    num_batches = 500
    LR = 0.0003
    datafolder = 3 #each number stores a single run of a game

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = TimeSformer(
    dim = 256,
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

    if os.path.exists('weights.pt'):
        print("loading weights do far ")
        model = torch.load('weights.pt')

    
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    model.to(device)

    d = deque(maxlen=num_prev_conditioning_frames)

    time.sleep(5) #provide time for user to position the mouse cursor
    y = 8

    for _ in range(200): #try out 100 model actions
        x = record_screen(D)
        x = np.array(x/255.0, dtype = np.float32).reshape((1,D,D))

        
        

        if len(d)==num_prev_conditioning_frames: #now the agent can act upon sufficient history
            obs = np.array(d)
            inp = torch.from_numpy(obs).view((batch_size, num_prev_conditioning_frames, C, D, D)).to(device)
            out = model(inp)
            probs = F.softmax(out, dim=-1) 

            act_idx = torch.argmax(probs)
            act_idx = torch.multinomial(probs, num_samples=1) # (B, 1)
            pred = act_idx.detach().cpu().numpy()[0][0] #should be a vector of length number of possible actions

            #pred = act_idx.detach().cpu().numpy() #should be a vector of length number of possible actions
            print("model pred ",pred)
            
            agent_action(pred)
            y = int(pred)

        #r = np.random.randint(0,9)
        #action_conditioning = (r/9.0)*np.ones((112,112), dtype = np.float32).reshape((1,D,D))
        #action_conditioning = np.random.sample((1,D,D)).astype(np.float32) #apply a random conditioning to each frame to help the network be able to think generatively
        
        #x = np.stack((x,action_conditioning)).reshape((2,D,D))
        #x = np.stack((x,x)).reshape((2,D,D))
        d.append(x)

    close_agent()



    



