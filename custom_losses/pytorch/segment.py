import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth = 1.):
    #ref- https://github.com/usuyama/pytorch-unet/blob/master/loss.py
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def seg_loss(pred, target, bce_weight=0.5):
    #ref- https://github.com/usuyama/pytorch-unet
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss

def crossentropy(pred,target):
    #print("pred shape ",pred.shape)
    #print("target shape ",target.shape)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) #0 is the unannotated class
    return criterion(pred,target)