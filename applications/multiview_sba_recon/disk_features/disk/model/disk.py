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


import torch
import numpy as np

#from disk_features.unets import Unet, thin_setup
#from disk_features.disk import NpArray, Features
#from disk_features.disk.model.detector import Detector

from applications.multiview_sba_recon.disk_features.unets import Unet, thin_setup
from applications.multiview_sba_recon.disk_features.disk import NpArray, Features
from applications.multiview_sba_recon.disk_features.disk.model.detector import Detector


DEFAULT_SETUP = {**thin_setup, 'bias': True, 'padding': True}

class DISK(torch.nn.Module):
    def __init__(
        self,
        desc_dim=128,
        window=8,
        setup=DEFAULT_SETUP,
        kernel_size=5,
    ):
        super(DISK, self).__init__()

        self.desc_dim = desc_dim
        self.unet = Unet(
            in_features=3, size=kernel_size,
            down=[16, 32, 64, 64, 64],
            up=[64, 64, 64, desc_dim+1],
            setup=setup,
        )
        self.detector = Detector(window=window)

    def _split(self, unet_output):
        '''
        Splits the raw Unet output into descriptors and detection heatmap.
        '''
        assert unet_output.shape[1] == self.desc_dim + 1

        descriptors = unet_output[:, :self.desc_dim]
        heatmap     = unet_output[:, self.desc_dim:]

        return descriptors, heatmap

    def features(
        self,
        images,
        kind='rng',
        **kwargs
    ) -> NpArray[Features]:
        ''' allowed values for `kind`:
            * rng
            * nms
        '''

        B = images.shape[0]
        try:
            descriptors, heatmaps = self._split(self.unet(images))
        except RuntimeError as e:
            if 'Trying to downsample' in str(e):
                msg = ('U-Net failed because the input is of wrong shape. With '
                       'a n-step U-Net (n == 4 by default), input images have '
                       'to have height and width as multiples of 2^n (16 by '
                       'default).')
                raise RuntimeError(msg) from e
            else:
                raise

        keypoints = {
            'rng': self.detector.sample,
            'nms': self.detector.nms,
        }[kind](heatmaps, **kwargs)

        features = []
        for i in range(B):
            features.append(keypoints[i].merge_with_descriptors(descriptors[i]))

        return np.array(features, dtype=object)
