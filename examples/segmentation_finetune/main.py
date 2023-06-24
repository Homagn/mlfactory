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


from models.pytorch.deeplabv3 import createDeepLabv3
from dataloaders.binary_seg_general import datahandler
from trainers.pytorch.train_seg2 import train_model


'''
import datahandler
from model import createDeepLabv3
from trainer import train_model
'''

batch_size = 4
epochs = 25
exp_directory = 'exp'
data_directory = '/datasets/small_segmentation/lidar_pillars'

def main(data_directory, exp_directory, epochs, batch_size):
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3()
    model.train()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    # Save the trained model
    torch.save(model, exp_directory / 'weights.pt')


if __name__ == "__main__":
    main(data_directory, exp_directory, epochs, batch_size)
