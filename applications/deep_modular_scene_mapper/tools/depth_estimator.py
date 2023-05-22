#pip install transformers
#use image size - (640,480)
import matplotlib
#apt-get install python3-tk
#pip install --user numba
#pip install scikit-image
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

import numpy as np
import open3d as o3d

from scipy import ndimage
import cv2

#import point_cloud_denoiser

def edges(d):
    #single derivative kind of does the job, adding double derivative does even better job

    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dx2 = ndimage.sobel(dx, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    dy2 = ndimage.sobel(dy, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)+ np.abs(dx2) + np.abs(dy2)


def reproject_depth_map(pcd):
    #========================================================================
    #depth reprojection part
    print("Now saving depth image reprojection of the pcd as depth.png...")
    img_width, img_height = (832, 448)
    #pcd = o3d.io.read_triangle_mesh('bunny.ply')
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)

    pcd.estimate_normals()
    pcd.normalize_normals()


    '''
    #apply perspective transformation of another camera pose, say the pose of that camera is trans, but you want 
    #to look at current view as if it was being viewed from that pose

    dtype = o3d.core.float32
    trans = np.array([[ 9.91051327e-01,  5.56196799e-02, -1.21341332e-01,
                        -1.64767893e-02],
                       [-5.60779272e-02,  9.98426329e-01, -3.62212664e-04,
                         1.46362140e-01],
                       [ 1.21130234e-01,  7.16354171e-03,  9.92610775e-01,
                        -7.51683117e-02],
                       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                         1.00000000e+00]])
    trans_inv = np.linalg.inv(trans)
    
    transformation = o3d.core.Tensor(trans_inv, dtype)
    pcd.transform(transformation)
    '''
    



    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'

    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
    renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
    #renderer_pc.scene.add_geometry("pcd", pcd, mat)
    renderer_pc.scene.add_geometry("__model__",pcd, mat)
    #renderer_pc.scene.add_geometry(pcd)

    print("current camera view matrix ",renderer_pc.scene.camera.get_view_matrix())




    depth_image = np.asarray(renderer_pc.render_to_depth_image())
    np.save('depth', depth_image)

    normalized_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    plt.imshow(normalized_image)
    plt.savefig('depth.png')



class monodepth(object):
    def __init__(self, vizpcd = False):
        print("Loading weights and initializing ...")
        self.feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
        self.model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
        self.vizpcd = vizpcd
        print("done")
    def predict_depth(self,imname):
        feature_extractor = self.feature_extractor
        model = self.model

        # load and resize the input image
        image = Image.open(imname)

        #doesnt work well when shadows in image or low resolution image
        #image = Image.open("../rgbd_mapping/data/indoor/rgb/0190.png")

        print("resizing image if needed ")
        new_height = 480 if image.height > 480 else image.height
        new_height -= (new_height % 32)
        new_width = int(new_height * image.width / image.height)
        diff = new_width % 32
        new_width = new_width - diff if diff < 16 else new_width + 32 - diff
        new_size = (new_width, new_height)
        image = image.resize(new_size)

        print("Inference using GLPN ...")
        # prepare image for the model
        inputs = feature_extractor(images=image, return_tensors="pt")

        # get the prediction from the model
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # remove borders
        pad = 16
        output = predicted_depth.squeeze().cpu().numpy() * 1000.0
        output = output[pad:-pad, pad:-pad]
        image = image.crop((pad, pad, image.width - pad, image.height - pad))


        


        width, height = image.size
        abs_max = np.max(output)

        print("max of depth image ",np.max(output))

        #depth_image = (output * 255 / np.max(output)).astype('uint8')
        depth_image = (output * 255 / np.max(output)) #depth image need not necessarily be normalized between 0 and 255, the max can be as well 500 or anything else

        #depth_image[depth_image>200.0]=0.0
        #hide sharp features in the depth image
        #depth_image[edges(depth_image) > 0.1*255.0] = 0.0  # Hide depth edges 


        image = np.array(image)
        '''

        
        '''


        if self.vizpcd:
            import project_rgbd
            # create rgbd image
            depth_o3d = o3d.geometry.Image(depth_image)
            image_o3d = o3d.geometry.Image(image)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

            # camera settings
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
            camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

            # create point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
            print("Done estimating pointcloud ")
            print("camera intrinsic width and height ",width, height)
            
            

            #both option 1 and 2  for visualization work well

            #option 1
            #o3d.visualization.draw_geometries([pcd])
            



            #option 2
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            coord_frame.scale(0.1, center=coord_frame.get_center()) #scale of the pcd becomes roughly around 30 units

            iphone_back_camera_params = {
                            "fx": 520.3,
                            "fy": 520.3,
                            "centerX": 320.0,
                            "centerY": 240.0,
                            "scalingFactor": 1
                        }
            pcd, pixel_points1 = project_rgbd.project_rgbd_to_pointcloud(image[:,:,::-1],depth_image, iphone_back_camera_params)

            project_rgbd.show_pcd([pcd,coord_frame])

            print("Now reprojecting depth map from another perspective ...")
            reproject_depth_map(pcd)

            


            

        return image, depth_image, abs_max



if __name__ == '__main__':
    md = monodepth(vizpcd = True)

    #md.predict_depth("IMG-1021.jpg")
    md.predict_depth("../extracted/0.jpg")