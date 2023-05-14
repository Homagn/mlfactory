import torch
import torch.nn.functional as F
from torchmetrics.functional import image_gradients # pip install torchmetrics
from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np

def depth_smoothness(pred,target):
    # Edges
    #https://torchmetrics.readthedocs.io/en/stable/image/image_gradients.html
    dy_true, dx_true = image_gradients(target)
    dy_pred, dx_pred = image_gradients(pred)

    weights_x = torch.exp(torch.mean(torch.abs(dx_true)))
    weights_y = torch.exp(torch.mean(torch.abs(dy_true)))


    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y

    depth_smoothness_loss = torch.mean(torch.abs(smoothness_x)) + torch.mean(
        torch.abs(smoothness_y)
    )
    return depth_smoothness_loss


def ssim_loss(pred,target):
    # Structural similarity (SSIM) index
    # https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html
    ssim = StructuralSimilarityIndexMeasure()
    ssim_loss = torch.mean(
        1
        - ssim(pred, target)
    )
    return ssim_loss

def contrast_difference_loss(pred,target):
    pm = torch.median(pred)
    tm = torch.median(target)
    
    high_diff = torch.abs( torch.mean(target[torch.where(target>tm)]) - torch.mean(pred[torch.where(pred>pm)]) )
    low_diff = torch.abs( torch.mean(target[torch.where(target<tm)]) - torch.mean(pred[torch.where(pred<pm)]) )

    return 0.5*(high_diff+low_diff)

def image_gradients_diff_loss(pred,target):
    # Edges
    #https://torchmetrics.readthedocs.io/en/stable/image/image_gradients.html
    dy_true, dx_true = image_gradients(target)
    dy_pred, dx_pred = image_gradients(pred)

    return 0.5* ( l1(dy_true,dy_pred) + l1(dx_true, dx_pred) )

def l1(pred,target):
    return torch.nn.L1Loss()(pred,target)


#----------------------------------------------------------------------------
#ref - https://github.com/wolverinn/Depth-Estimation-PyTorch/blob/master/fyn_main.py
class GradLoss(torch.nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    # L1 norm
    def forward(self, grad_fake, grad_real):
        return torch.mean( torch.abs(grad_real-grad_fake) )

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = torch.nn.Parameter(weight)
    grad_x = conv1(img)
    # grad y
    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = torch.nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x
def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)
#----------------------------------------------------------------------------




'''
def loss(pred, target): #tried to define my own fancy loss
    pred = F.relu(pred)
    #in training depth losses, sometimes sensors give hard edges around objects
    #model should ignore those otherwise training will be difficult
    target = torch.where(target==0.0, pred, target)

    #produces interesting shapes but nothing other than that
    #return 0.3*ssim_loss(pred,target) + 0.3*contrast_difference_loss(pred,target) + 0.3*image_gradients_diff_loss(pred,target)

    return 0.3*torch.mean(torch.abs(target - pred)) + 0.3*ssim_loss(pred, target) + 0.3*contrast_difference_loss(pred,target)

'''

def loss(pred, target):
    #print("mean pred ",torch.mean(pred))
    #print("max pred ",torch.max(pred))

    grad_criterion = GradLoss()
    grad_real, grad_fake = imgrad_yx(target), imgrad_yx(pred)
    loss = torch.mean( torch.abs(grad_real-grad_fake) )
    return loss
