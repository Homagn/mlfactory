import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
import cv2

batch_size = 2
kwargs = {'num_workers': 4, 'pin_memory': True} 

'''
cifar_transform = transforms.Compose([
            transforms.Resize(size = (64,64)),
            transforms.Grayscale(),
            transforms.ToTensor(),
    ])
'''

#can use center crop to crop the center part of the image

'''
cifar_transform = transforms.Compose([
            transforms.CenterCrop( (64,64)),
            transforms.Grayscale(),
            transforms.ToTensor(),
    ])
'''

cifar_transform = transforms.Compose([
            transforms.CenterCrop( (128,128)),
            transforms.Resize(size = (64,64)),
            transforms.Grayscale(),
            transforms.ToTensor(),
    ])

#cifar_trainset = datasets.SVHN(root='/datasets', download=True, transform=cifar_transform)
#cifar_trainset = datasets.CIFAR10(root='/datasets', download=True, transform=cifar_transform)
#cifar_trainset = datasets.FashionMNIST(root='/datasets', download=True, transform=cifar_transform)
#cifar_trainset = datasets.STL10(root='/datasets', download=True, transform=cifar_transform)
#cifar_trainset = datasets.CelebA(root='/datasets', download=True, transform=cifar_transform)
#cifar_trainset = datasets.MNIST(root='/datasets', download=True, transform=cifar_transform)

cifar_trainset = datasets.ImageFolder(root='/datasets/behavior_cloning/maze_game/sampled', transform=cifar_transform)

train_loader = DataLoader(dataset=cifar_trainset, batch_size=batch_size, shuffle=True, **kwargs)


for batch_idx, (x, _) in enumerate(train_loader):
    x = x.permute(0,2,3,1)
    #x = x.permute((-1,1,2,0))
    x = x.cpu().numpy()
    cv2.imshow("sample ",x[0].reshape((64,64,1)))
    cv2.waitKey(0)
    sys.exit(0)
