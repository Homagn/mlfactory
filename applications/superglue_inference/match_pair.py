import os
import sys



# An installation agnostic method to find and link to root of the package which is mlfactory
#==========================================================
import re

try: #testing the functions locally without pip install
    import __init__
    cimportpath = os.path.abspath(__init__.__file__)
    if 'extensions' in cimportpath:
        print("Non local usage")
        import mlfactory
        cimportpath = os.path.abspath(mlfactory.__file__)+'/applications/superglue_inference/__init__.py'

except: #testing while mlfactory is installed using pip
    print("Non local usage")
    import mlfactory
    cimportpath = os.path.abspath(mlfactory.__file__)+'/applications/superglue_inference/__init__.py'

idxlist = [m.start() for m in re.finditer(r"/", cimportpath)]
invoking_submodule = cimportpath[idxlist[-2]+1:idxlist[-1]]
print("In superglue_inference/match_pair.py got invoking submodule using re",invoking_submodule)
main_package_loc = cimportpath[:cimportpath.rfind('mlfactory')+len('mlfactory')]
print("In superglue_inference/match_pair.py got main package location ",main_package_loc)


os.environ['superglue'] = main_package_loc+'/applications/superglue_inference'
os.environ['top'] = main_package_loc
sys.path.append(os.path.join(os.environ['superglue']))
sys.path.append(os.path.join(os.environ['top']))
#==========================================================


from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch

#import superglue_inference
from applications.superglue_inference.models.matching import Matching
from applications.superglue_inference.models.utils import (AverageTimer, VideoStreamer,make_matching_plot_fast, frame2tensor)

#from models.matching import Matching
#from models.utils import (AverageTimer, VideoStreamer,make_matching_plot_fast, frame2tensor)


import numpy as np
#torch.set_grad_enabled(False)

from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform

def check_weights():
    gpath = os.environ['superglue']
    if os.path.exists(gpath+"/models/weights")==False:
    #if os.path.exists("models/weights")==False:
        print("pretrained weights need to be downloaded and extracted ")
        os.system("pip install gdown")
        os.system("gdown --id 1RU5jIDkvaXE_h0fEew-xYCiJs87QDMEG")
        os.system("unzip superglueweights.zip -d"+gpath+"/models/")
        print("done extracting !")
        os.system("rm -rf superglueweights.zip")


class matcher(object):
    def __init__(self, reshape_size = (256,256)):
        check_weights()
        self.reshape_size = reshape_size

        force_cpu = False
        nms_radius = 4
        keypoint_threshold = 0.005
        max_keypoints = -1 #detect as many as possible
        superglue = 'indoor'
        sinkhorn_iterations = 20
        match_threshold = 0.2
        show_keypoints = False
        self.viz_plt = False



        device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
        print('Running inference on device \"{}\"'.format(device))
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        self.matching = Matching(config).eval().to(device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        self.device = device
        self.show_keypoints = show_keypoints

    def prepare_images(self, img1, img2):
        #from PIL import Image, ImageOps
        def rgb2gray(rgb):

            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

            return gray

        '''
        print("feature matcher prepare images input shapes ",img1.shape, img2.shape)
        cv2.imshow("img1 ",img1)
        cv2.waitKey(0)
        cv2.imshow("img2 ",img2)
        cv2.waitKey(0)
        '''
        if len(img1.shape)>2:
            #print("changing to grayscale")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #for some unknown reason this creates seg fault
            #img1 =  ImageOps.grayscale(img1)
            #img1 = rgb2gray(img1)
        if len(img2.shape)>2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            #img2 =  ImageOps.grayscale(img2)
            #img2 = rgb2gray(img2)
        return np.array(img1), np.array(img2)

    def prepare_images_depth(self, filename1, filename2):
        img1 = cv2.imread(filename1,0)          # queryImage
        img2 = cv2.imread(filename2,0)          # trainImage

        width = img1.shape[1]

        img1 = img1[:,width//2:]
        img2 = img2[:,width//2:]

        res_shape = (256,256)

        img1 = cv2.resize(img1,res_shape)
        img2 = cv2.resize(img2,res_shape)

        return np.array(img1), np.array(img2)

    def match(self,im1, im2, viz = True, ransac_refine = False):
        torch.set_grad_enabled(False)
        #frame = cv2.imread("assets/longfellow/000190_left.png",0)
        #frame = np.array(frame)
        frame = im1

        frame_tensor = frame2tensor(frame, self.device)
        last_data = self.matching.superpoint({'image': frame_tensor})
        last_data = {k+'0': last_data[k] for k in self.keys}
        last_data['image0'] = frame_tensor
        last_frame = frame
        last_image_id = 0


        #frame = cv2.imread("assets/longfellow/000200_left.png",0)
        #frame = np.array(frame)
        frame = im2
        #print("frame type and shape ",type(frame), frame.shape)

        #stem0, stem1 = last_image_id, vs.i - 1
        stem0, stem1 = 0,1

        frame_tensor = frame2tensor(frame, self.device)
        pred = self.matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()

        #print("kpts 0 ",kpts0)
        #print("kpts 1 ",kpts1)


        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = self.matching.superpoint.config['keypoint_threshold']
        m_thresh = self.matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        '''
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=self.show_keypoints, small_text=small_text)
        '''

        

        if ransac_refine:
            #can try projective transform ransac (not always helpful)
            
            #print("matching confidence before ransac ",confidence[valid])

            model, inliers = ransac(
                (mkpts0, mkpts1),
                ProjectiveTransform, min_samples=4,
                residual_threshold=8, max_trials=100
            )
            #print("matching confidences after ransac ",confidence[valid][inliers])
            





        #if not opt.no_display:
        if viz:
            if ransac_refine:
                out = make_matching_plot_fast(
                    last_frame, frame, kpts0, kpts1, mkpts0[inliers], mkpts1[inliers], color, text,
                    path=None, show_keypoints=self.show_keypoints, small_text=small_text)
            else:
                out = make_matching_plot_fast(
                    last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
                    path=None, show_keypoints=self.show_keypoints, small_text=small_text)

            if self.viz_plt:
                import matplotlib
                #matplotlib.use('TkAgg')
                from matplotlib import pyplot as plt
                fig, ax = plt.subplots()
                ax.set_axis_off()
                ax.imshow(out)
                #plt.show()
            else:
                cv2.imshow('SuperGlue matches', out)
                cv2.waitKey(0)



        torch.set_grad_enabled(True)




        return mkpts0, mkpts1



if __name__ == '__main__':
    m = matcher()
    #m.viz_plt = True
    im1 = np.array(cv2.imread("sample1.png",0))
    im2 = np.array(cv2.imread("sample2.png",0))
    m.match(im1,im2)





