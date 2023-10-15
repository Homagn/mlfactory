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

#from disk_features.feature import extract_features, match_features
from applications.multiview_sba_recon.disk_features.feature import extract_features, match_features


import os
import cv2
import numpy as np
from tqdm import tqdm
#import exifread

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix



class Camera:
    def __init__(self, id, img, kp, desc, match2d3d):
        self.id = id
        self.img = img
        self.kp = kp
        self.desc = desc 
        self.match2d3d = match2d3d
        self.Rt = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.reconstruct = False

    def setRt(self, R, t):
        self.Rt = np.hstack((R, t))
        self.reconstruct = True
    
    def getRt(self):
        return self.Rt[:3,:3], self.Rt[:3, 3]

    def getRelativeRt(self, cam2):
        return cam2.Rt[:3,:3].T.dot(self.Rt[:3,:3]), cam2.Rt[:3, :3].T.dot(self.Rt[:3, 3] - cam2.Rt[:3, 3])
    
    def getP(self, K):
        return np.matmul(K, self.Rt)
    
    def getPos(self):
        pts = np.array([[0,0,0]]).T
        pts = self.Rt[:3,:3].T.dot(pts)- self.Rt[:3,3][:,np.newaxis]
        return pts[:,0]
    
    def getFeature(self):
        return (self.kp, self.desc)

def get_image_params(images_dir):
    K = []
    h, w, c = cv2.imread(images_dir + os.listdir(images_dir)[1]).shape
    image_width, image_height = (w, h) if w > h else (h, w)

    return {'width': image_width, 'height': image_height}

def triangulate(cam1, cam2, idx0, idx1, K, point_cloud, point_color):
    points_3d = cv2.triangulatePoints(cam1.getP(K), cam2.getP(K), cam1.kp[idx0].T, cam2.kp[idx1].T)
    points_3d = points_3d / points_3d[3]
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    points_3d = points_3d[:, 0, :]
    point2d_ind = idx1[np.where(cam1.match2d3d[idx0] ==  -1)]
    for w, i in enumerate(idx0):
        if cam1.match2d3d[i] == -1:
            point_cloud.append(points_3d[w])
            point_color.append(cam1.img[int(cam1.kp[i][1]), int(cam1.kp[i][0]), :])
            cam1.match2d3d[i] = len(point_cloud) - 1
        cam2.match2d3d[idx1[w]] = cam1.match2d3d[i]
    point3d_ind = cam2.match2d3d[point2d_ind]
    x = np.hstack((cv2.Rodrigues(cam2.getRt()[0])[0].ravel(), cam2.getRt()[1].ravel(), np.array(point_cloud)[point3d_ind].ravel()))
    A = ba_sparse(point3d_ind, x)
    res = least_squares(calculate_reprojection_error, x, jac_sparsity=A, x_scale='jac', ftol=1e-8, args=(K, cam2.kp[point2d_ind]))
    R, t, point_3D = cv2.Rodrigues(res.x[:3])[0], res.x[3:6], res.x[6:].reshape((len(point3d_ind), 3))
    for i, j in enumerate(point3d_ind): point_cloud[j] = point_3D[i]
    cam2.setRt(R, t.reshape((3,1)))
    return point_cloud, point_color

def to_ply(img_dir, point_cloud, colors, subfix = "_sparse.ply"):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(out_colors.shape, out_points.shape)
    verts = np.hstack([out_points, out_colors])
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]

    ####################################################
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    xyz = verts[:,:3]
    colors = verts[:,3:] /255.0
    colorr = colors.copy()
    colorr[:,0] = colors[:,2]
    colorr[:,1] = colors[:,1]
    colorr[:,2] = colors[:,0]


    print("number of points ",xyz.shape[0])

    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colorr)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #vis.create_window(width=832, height=448)
    

    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # run visualizer main loop
    print("Press Q or Excape to exit")
    vis.run()
    

    vis.destroy_window()

    ####################################################


    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
    print(img_dir + '/Point_Cloud/' + img_dir.split('/')[-2] + subfix)
    if not os.path.exists(img_dir + '/Point_Cloud/'):
        os.makedirs(img_dir + '/Point_Cloud/')
    with open(img_dir + '/Point_Cloud/' + img_dir.split('/')[-2] + subfix, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

def ba_sparse(point3d_ind, x):
    A = lil_matrix((len(point3d_ind)*2, len(x)), dtype=int)
    A[np.arange(len(point3d_ind)*2), :6] = 1
    for i in range(3):
        A[np.arange(len(point3d_ind))*2, 6 + np.arange(len(point3d_ind))*3 + i] = 1
        A[np.arange(len(point3d_ind))*2 + 1, 6 + np.arange(len(point3d_ind))*3 + i] = 1
    return A

def calculate_reprojection_error(x, K, point_2D):
    R, t, point_3D = x[:3], x[3:6], x[6:].reshape((len(point_2D), 3))
    reprojected_point, _ = cv2.projectPoints(point_3D, R, t, K, distCoeffs=None)
    reprojected_point = reprojected_point[:, 0, :]
    return (point_2D - reprojected_point).ravel()


def register_images(image_folder, ordered_images, image_params, intrinsics, cameras, point_cloud, point_color):
    images = ordered_images
    K = intrinsics
    iparams = image_params
    img_dir = image_folder


    j = 0
    for i in tqdm(range(len(images))):
        if images[i].split('.')[-1] in ['JPG', 'jpg', 'PNG', 'png', 'RAW', 'raw', 'TIF', 'tif']:
            img = cv2.imread(img_dir + images[i])
            if img.shape[1] != iparams['width'] or img.shape[0] != iparams['height']:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            kp, des = extract_features(img)
            cv2.imshow("image ",img)
            cv2.waitKey(1)
            cameras.append(Camera(images[i], img.copy(), kp, des, np.ones((len(kp),), dtype='int32')*-1))
            if j > 0:
                pts0_, pts1_, idx0, idx1 = match_features(cameras[j-1], cameras[j])
                E, mask = cv2.findEssentialMat(pts0_, pts1_, K, method=cv2.RANSAC, prob=0.999, threshold=1)
                idx0, idx1 = idx0[mask.ravel() == 1], idx1[mask.ravel() == 1]
                _, R, t, _ = cv2.recoverPose(E, pts0_[mask.ravel() == 1], pts1_[mask.ravel() == 1], K)
                if j != 1:
                    match = np.int32(np.where(cameras[j-1].match2d3d[idx0] != -1)[0])
                    if len(match) < 8: continue
                    ret, rvecs, t, inliers = cv2.solvePnPRansac(np.float32(point_cloud)[cameras[j-1].match2d3d[idx0[match]]], cameras[j].kp[idx1[match]], K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
                    R, _ = cv2.Rodrigues(rvecs)
                cameras[j].setRt(R, t)
                point_cloud, point_color =  triangulate(cameras[j-1], cameras[j], idx0, idx1, K, point_cloud, point_color)
            j += 1

    to_ply(img_dir, np.array(point_cloud), np.array(point_color))
    to_ply(img_dir, np.array([cam.getPos() for cam in cameras]), np.ones_like(np.array([cam.getPos() for cam in cameras]))*0, '_campos.ply')





if __name__ == '__main__':
    img_dir = "/datasets/reconstruction_examples/GustavIIAdolf/"
    #img_dir = "/datasets/reconstruction_examples/castle-P19/"
    #img_dir = "/datasets/reconstruction_examples/fountain-P11/"
    #img_dir = "/datasets/reconstruction_examples/pedestal_fan/" #custom data (iphone portrait) works average because baseline distances are low
    #img_dir = "/datasets/reconstruction_examples/cambridge_office/" #custom data (iphone landscape) works well because baseline distances (camera translation vector difference between consequtive frames) are high

    iparams = get_image_params(img_dir)
    images = sorted( filter( lambda x: os.path.isfile(os.path.join(img_dir, x)), os.listdir(img_dir) ) )
    
    cameras = []
    point_cloud = []
    point_color = []

    
    #Gustav
    K = np.array([[ 2.39395217e+03 ,-3.41060513e-13,  9.32382177e+02],
                 [ 0.00000000e+00,  2.39811854e+03,  6.28264995e+02],
                 [ 0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])
    

    '''
    #castle-P19
    K = np.array([[ 2759.48 , 0.0 ,  1520.69],
                 [ 0.00000000e+00,  2764.16,  1006.81 ],
                 [ 0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])
    '''

    '''
    #fountain-P11
    K = np.array([[ 2759.48 , 0.0 ,  1520.69],
                 [ 0.00000000e+00,  2764.16,  1006.81 ],
                 [ 0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])
    '''

    '''
    #Iphone 11 portrait photo
    K = np.array( [[3.20512987e+03, 0.00000000e+00, 1.99443897e+03],
                 [0.00000000e+00, 3.17391061e+03, 1.41309060e+03],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]] )
    '''

    '''
    #iphone 11 landscape video (cambridge_office)
    K = np.array([[3.20512987e+03, 0.00000000e+00, 960.0],
                 [0.00000000e+00, 3.17391061e+03, 540.0],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    '''


    K = np.array(K, dtype=float)
    

    register_images(img_dir, images, iparams, K, cameras, point_cloud, point_color )


    '''
    j = 0
    for i in tqdm(range(len(images))):
        if images[i].split('.')[-1] in ['JPG', 'jpg', 'PNG', 'png', 'RAW', 'raw', 'TIF', 'tif']:
            img = cv2.imread(img_dir + images[i])
            if img.shape[1] != iparams['width'] or img.shape[0] != iparams['height']:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            kp, des = extract_features(img)
            cv2.imshow("image ",img)
            cv2.waitKey(1)
            cameras.append(Camera(images[i], img.copy(), kp, des, np.ones((len(kp),), dtype='int32')*-1))
            if j > 0:
                pts0_, pts1_, idx0, idx1 = match_features(cameras[j-1], cameras[j])
                E, mask = cv2.findEssentialMat(pts0_, pts1_, K, method=cv2.RANSAC, prob=0.999, threshold=1)
                idx0, idx1 = idx0[mask.ravel() == 1], idx1[mask.ravel() == 1]
                _, R, t, _ = cv2.recoverPose(E, pts0_[mask.ravel() == 1], pts1_[mask.ravel() == 1], K)
                if j != 1:
                    match = np.int32(np.where(cameras[j-1].match2d3d[idx0] != -1)[0])
                    if len(match) < 8: continue
                    ret, rvecs, t, inliers = cv2.solvePnPRansac(np.float32(point_cloud)[cameras[j-1].match2d3d[idx0[match]]], cameras[j].kp[idx1[match]], K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
                    R, _ = cv2.Rodrigues(rvecs)
                cameras[j].setRt(R, t)
                point_cloud, point_color =  triangulate(cameras[j-1], cameras[j], idx0, idx1, K, point_cloud, point_color)
            j += 1

    to_ply(img_dir, np.array(point_cloud), np.array(point_color))
    to_ply(img_dir, np.array([cam.getPos() for cam in cameras]), np.ones_like(np.array([cam.getPos() for cam in cameras]))*0, '_campos.ply')
    '''