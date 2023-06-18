import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os, sys, copy, time
import numpy as np
import cv2 
import random as rand
from datetime import datetime as dt



#import seaborn as sns
import math, glob
from scipy import stats  

# An installation agnostic method to find and link to root of the package which is mlfactory
#==========================================================
import re

try: #testing the functions locally without pip install
    import __init__
    cimportpath = os.path.abspath(__init__.__file__)
    if 'extensions' in cimportpath:
        print("Non local usage")
        import mlfactory
        cimportpath = os.path.abspath(mlfactory.__file__)+'/applications/visual_odometry/__init__.py'

except: #testing while mlfactory is installed using pip
    print("Non local usage")
    import mlfactory
    cimportpath = os.path.abspath(mlfactory.__file__)+'/applications/visual_odometry/__init__.py'

idxlist = [m.start() for m in re.finditer(r"/", cimportpath)]
invoking_submodule = cimportpath[idxlist[-2]+1:idxlist[-1]]
print("In visual_odometry/sfm.py got invoking submodule using re",invoking_submodule)
main_package_loc = cimportpath[:cimportpath.rfind('mlfactory')+len('mlfactory')]
print("In visual_odometry/sfm.py got main package location ",main_package_loc)


os.environ['sfm'] = main_package_loc+'/applications/visual_odometry'
os.environ['top'] = main_package_loc
sys.path.append(os.path.join(os.environ['sfm']))
sys.path.append(os.path.join(os.environ['top']))
#==========================================================


#import superglue based matcher (superior to sift)
from applications.superglue_inference import match_pair
from applications.deep_modular_scene_mapper.tools import project_rgbd


import open3d as o3d




# =============================================================================================================================================================================================================================== #
# -----> Object for Zhang's Grid <----- #
# =============================================================================================================================================================================================================================== #
class Cells:
    def __init__(self):
        self.pts = list()
        self.pairs = dict()

    def rand_pt(self):
        return rand.choice(self.pts)
class VisualOdometry:
    def __init__(self):
        self.m = match_pair.matcher()
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
    # =========================================================================================================================================================================================================================== #
    # Get Random 8 points from different regions in a Image using Zhang's 8x8 Grid
    # =========================================================================================================================================================================================================================== #
    def get_rand8(self, grid: np.array)-> list:            
        cells = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        rand_grid_index = rand.choices(cells, k = 8)   
        rand8 = list() 
        rand8_ = list()        
        for index in rand_grid_index:
            if grid[index].pts: 
                pt = grid[index].rand_pt()
                rand8.append(pt)
            else:
                index = rand.choice(cells)
                while not grid[index].pts or index in rand_grid_index:
                    index = rand.choice(cells) 
                pt = grid[index].rand_pt()
                rand8.append(pt)

            # -----> find the correspondence given point <----- #
            rand8_.append(grid[index].pairs[pt])
        return rand8, rand8_
    # =========================================================================================================================================================================================================================== #
    # Calculate Fundamental Matrix for the given * points from RANSAC
    # =========================================================================================================================================================================================================================== #  
    def calculate_fundamental_matrix(self, pts_cf: np.array, pts_nf: np.array, method = "8point")-> list:
        if method=="8point":
            F_cv2,_ = cv2.findFundamentalMat(pts_cf,pts_nf,cv2.FM_8POINT)
        elif method=="ransac":
            F_cv2,_ = cv2.findFundamentalMat(pts_cf,pts_nf,cv2.RANSAC)
        else:
            F_cv2,_ = cv2.findFundamentalMat(pts_cf,pts_nf,cv2.FM_LMEDS)


        #F_cv2,_ = cv2.findFundamentalMat(pts_cf,pts_nf,cv2.FM_LMEDS)
        #F_cv2,_ = cv2.findFundamentalMat(pts_cf,pts_nf,cv2.RANSAC)

        mat = []
        origin = [0.,0.]
        origin_ = [0.,0.]   
        origin = np.mean(pts_cf, axis = 0)
        origin_ = np.mean(pts_nf, axis = 0) 
        k = np.mean(np.sum((pts_cf - origin)**2 , axis=1, keepdims=True)**.5)
        k_ = np.mean(np.sum((pts_nf - origin_)**2 , axis=1, keepdims=True)**.5)
        k = np.sqrt(2.)/k
        k_ = np.sqrt(2.)/k_
        x = ( pts_cf[:, 0].reshape((-1,1)) - origin[0])*k
        y = ( pts_cf[:, 1].reshape((-1,1)) - origin[1])*k
        x_ = ( pts_nf[:, 0].reshape((-1,1)) - origin_[0])*k_
        y_ = ( pts_nf[:, 1].reshape((-1,1)) - origin_[1])*k_
        A = np.hstack((x_*x, x_*y, x_, y_ * x, y_ * y, y_, x,  y, np.ones((len(x),1)))) 
        U,S,V = np.linalg.svd(A)
        F = V[-1]
        F = np.reshape(F,(3,3))
        U,S,V = np.linalg.svd(F)
        S[2] = 0
        F = U@np.diag(S)@V  
        T1 = np.array([[k, 0,-k*origin[0]], [0, k, -k*origin[1]], [0, 0, 1]])
        T2 = np.array([[k_, 0,-k_*origin_[0]], [0, k_, -k_*origin_[1]], [0, 0, 1]])
        F = T2.T @ F @ T1
        F = F / F[-1,-1]
        return F,F_cv2

    # =========================================================================================================================================================================================================================== #
    # Estimate Fundamental Matrix from the given correspondences using RANSAC
    # =========================================================================================================================================================================================================================== #  
    def estimate_fundamental_matrix_RANSAC(self, pts1, pts2, grid, epsilon = 0.05, tries = 10)-> list:
        max_inliers= 0
        F_best = []
        S_in = []
        confidence = 0.99
        N = tries #sys.maxsize
        count = 0
        while N > count:
            S = []
            counter = 0
            x_1,x_2 = self.get_rand8(grid)
            F,F_b = self.calculate_fundamental_matrix(np.array(pts1), np.array(pts2))
            ones = np.ones((len(pts1),1))
            x = np.hstack((pts1,ones))
            x_ = np.hstack((pts2,ones))
            e, e_ = x @ F.T, x_ @ F
            error = np.sum(e_* x, axis = 1, keepdims=True)**2 / np.sum(np.hstack((e[:, :-1],e_[:,:-1]))**2, axis = 1, keepdims=True)
            inliers = error<=epsilon
            counter = np.sum(inliers)
            if max_inliers <  counter:
                max_inliers = counter
                F_best = F 
            I_O_ratio = counter/len(pts1)
            if np.log(1-(I_O_ratio**8)) == 0: continue
            N = np.log(1-confidence)/np.log(1-(I_O_ratio**8))
            count += 1
        return F_best
    # =========================================================================================================================================================================================================================== #
    # Estimate Essential Matrix 
    # =========================================================================================================================================================================================================================== #
    def estimate_Essential_Matrix(self, K: np.array, F: np.array)-> np.array:   
        E = K.T @ F @ K
        U,S,V = np.linalg.svd(E)
        S = [[1,0,0],[0,1,0],[0,0,0]]
        E = U @ S @ V
        return E

    # =========================================================================================================================================================================================================================== #
    # Perform Linear Triangulation
    # =========================================================================================================================================================================================================================== #
    def linear_triangulation(self, K: np.array, C1: np.array, R1: np.array, C2: np.array, R2: np.array, pt: np.array, pt_: np.array)-> list:
        P1 = K @ np.hstack((R1, -R1 @ C1))
        P2 = K @ np.hstack((R2, -R2 @ C2))  
        X = []
        for i in range(len(pt)):
            x1 = pt[i]
            x2 = pt_[i]
            A1 = x1[0]*P1[2,:]-P1[0,:]
            A2 = x1[1]*P1[2,:]-P1[1,:]
            A3 = x2[0]*P2[2,:]-P2[0,:]
            A4 = x2[1]*P2[2,:]-P2[1,:]      
            A = [A1, A2, A3, A4]
            U,S,V = np.linalg.svd(A)
            V = V[3]
            V = V/V[-1]
            X.append(V)
        #print("linear traingulation ",X)
        return X

    # =========================================================================================================================================================================================================================== #
    # Estimate the camera Pose
    # =========================================================================================================================================================================================================================== #
    def camera_pose(self, K: np.array, E: np.array):
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        U,S,V = np.linalg.svd(E)
        poses = {}
        poses['C1'] = U[:,2].reshape(3,1)
        poses['C2'] = -U[:,2].reshape(3,1)
        poses['C3'] = U[:,2].reshape(3,1)
        poses['C4'] = -U[:,2].reshape(3,1)
        poses['R1'] = U @ W @ V
        poses['R2'] = U @ W @ V 
        poses['R3'] = U @ W.T @ V
        poses['R4'] = U @ W.T @ V
        for i in range(4):
            C = poses['C'+str(i+1)]
            R = poses['R'+str(i+1)]
            if np.linalg.det(R) < 0:
                C = -C 
                R = -R 
                poses['C'+str(i+1)] = C 
                poses['R'+str(i+1)] = R
            I = np.eye(3,3)
            M = np.hstack((I,C.reshape(3,1)))
            poses['P'+str(i+1)] = K @ R @ M
        return poses

    # =========================================================================================================================================================================================================================== #
    # Find the Rotation and Translation parametters
    # =========================================================================================================================================================================================================================== #
    def extract_Rot_and_Trans(self, R1: np.array, t: np.array, pt: np.array, pt_: np.array, K: np.array):
        C = [[0],[0],[0]]
        R = np.eye(3,3)
        P = np.eye(3,4)
        P_ = np.hstack((R1,t))
        X1 = self.linear_triangulation(K, C, R,t,R1, pt, pt_)
        X1 = np.array(X1)   
        count = 0
        for i in range(X1.shape[0]):
            x = X1[i,:].reshape(-1,1)
            if R1[2]@np.subtract(x[0:3],t) > 0 and x[2] > 0: count += 1
        return count

    def extract_correspondences(self, current_frame, next_frame, use_superglue):
        m = self.m
        sift = self.sift
        bf = self.bf
        if use_superglue==False:
            #current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) 
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)  
            kp_cf,des_current = sift.detectAndCompute(current_frame,None)
            kp_nf,des_next = sift.detectAndCompute(next_frame,None)


            # -----> Extract the best matches <----- #
            best_matches = []
            
            matches = bf.knnMatch(des_current,des_next,k=2)
            for m,n in matches:
                if m.distance < 0.9*n.distance: best_matches.append(m)
            
            # -----> Initialise the grids and points array variables <----- #
            point_correspondence_cf = np.zeros((len(best_matches),2))
            point_correspondence_nf = np.zeros((len(best_matches),2))
            grid = np.empty((8,8), dtype=object)
            grid[:,:] = Cells()

            # ----> Generating Zhang's Grid & extracting points from matches<----- #
            for i, match in enumerate(best_matches):
                '''
                j = int(kp_cf[match.queryIdx].pt[0]/x_bar)
                k = int(kp_cf[match.queryIdx].pt[1]/y_bar)
                grid[j,k].pts.append(kp_cf[match.queryIdx].pt)
                grid[j,k].pairs[kp_cf[match.queryIdx].pt] = kp_nf[match.trainIdx].pt
                '''

                point_correspondence_cf[i] = kp_cf[match.queryIdx].pt[0], kp_cf[match.queryIdx].pt[1]
                point_correspondence_nf[i] = kp_nf[match.trainIdx].pt[0], kp_nf[match.trainIdx].pt[1]
            #F = func.estimate_fundamental_matrix_RANSAC(point_correspondence_cf, point_correspondence_nf, grid, 0.05)                  # Estimate the Fundamental matrix # 

        else:

            i1,i2 = m.prepare_images(current_frame,next_frame)
            point_correspondence_cf,point_correspondence_nf = m.match(i1,i2, viz = False, ransac_refine = True)
            
            #downsample correspondences by half to increase speed
            '''
            chosen = np.random.choice(point_correspondence_cf.shape[0], point_correspondence_cf.shape[0]//2, replace = False)
            point_correspondence_cf = point_correspondence_cf[chosen]
            point_correspondence_nf = point_correspondence_nf[chosen]
            '''

            #print("point_correspondence_cf shape ",point_correspondence_cf.shape)
        #print("point_correspondence_cf shape ",point_correspondence_cf.shape)
        return point_correspondence_cf, point_correspondence_nf

    def find_pose_change(self, K, point_correspondence_cf, point_correspondence_nf, essential_estimation):
        if essential_estimation=="cv":
            E, mask = cv2.findEssentialMat(point_correspondence_cf, point_correspondence_nf, K, cv2.RANSAC, 0.999, 1.0);
            #E, mask = cv2.findEssentialMat(point_correspondence_cf, point_correspondence_nf, K, cv2.RANSAC, 0.5, 1.0)
        
        elif essential_estimation=="svd":
            F,_ = self.calculate_fundamental_matrix(point_correspondence_cf, point_correspondence_nf, method = "ransac")
            E = self.estimate_Essential_Matrix(K, F)       

        else:
            F = self.estimate_fundamental_matrix_RANSAC(point_correspondence_cf, point_correspondence_nf, grid, 0.05)    
            E = self.estimate_Essential_Matrix(K, F)                                                                                     # Estimate the Essential Matrix #
        

        
        pose = self.camera_pose(K,E)                                                                                                        # Estimate the Posses Matrix #
        # -----> Estimate Rotationa and Translation points <----- #
        flag = 0
        for p in range(4):
            R = pose['R'+str(p+1)]
            T = pose['C'+str(p+1)]
            Z = self.extract_Rot_and_Trans(R, T, point_correspondence_cf, point_correspondence_nf, K)
            if flag < Z: flag, reg = Z, str(p+1)

        R = pose['R'+reg]
        t = pose['C'+reg]
        if t[2] < 0: t = -t

        return R, t

    def accumulate_poses(self, Translation, Rotation, t, R, pose_sequence):
        x_cf = Translation[0]
        z_cf = Translation[2]
        Translation += Rotation.dot(t)
        Rotation = R.dot(Rotation)
        x_nf = Translation[0]
        z_nf = Translation[2]




        #====================================
        #append new pose to the pose sequence
        T = np.eye(4)
        T[:3, :3] = Rotation
        T[0, 3] = Translation[0][0]
        T[1, 3] = Translation[1][0]
        T[2, 3] = Translation[2][0]
        new_pose = T
        pose_sequence.append(new_pose)
        #====================================
        return Translation, Rotation, pose_sequence

    def visualize_trajectory(self,pose_sequence, every = 10, size = 10, display_result = ''):
        ######################################
        #show all the coordinate transformations
        coord_meshes = []
        for i in range(len(pose_sequence)):
            #self.pose_sequence[i] = np.dot(self.pose_sequence[i], self.refined_poses[i])

            coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            #coord_mesh = o3d.geometry.TriangleMesh.create_arrow()
            #coord_mesh = o3d.geometry.TriangleMesh.create_cone()
            if i==0:
                #make the first one big to identify easily
                coord_mesh.scale(size*2, center=coord_mesh.get_center()) 
            else:
                coord_mesh.scale(size, center=coord_mesh.get_center()) 

            coord_mesh.transform(pose_sequence[i])
            if i%every==0:
                coord_meshes.append(coord_mesh)
        if display_result=='colab':
            project_rgbd.show_pcd_colab(coord_meshes)
        else:
            project_rgbd.show_pcd(coord_meshes)

        ######################################



    # =========================================================================================================================================================================================================================== #

def process_resize(frame, image_width, image_height): #resize according to the size in which camera was calibrated
    frame = cv2.resize(frame, (image_width, image_height))
    return frame





if __name__=="__main__":
    #use superglue False and "cv" is the fastest but often innacurate
    #Current calibration settings work well with iphone11 back camera shooting video in portrait mode

    use_superglue = False #uses sift which produce lesser matches but is much faster
    triangulate_points = False
    essential_estimation = "cv" #"cv", "zhang", "svd" 
    image_width, image_height = 320, 240 #algorithm is slower for larger image sizes, more inaccurate for lower sizes
    # =============================================================================================================================================================================================================================== #
    # -----> Alias <----- #
    # =============================================================================================================================================================================================================================== #
    inv = np.linalg.inv
    det = np.linalg.det
    svd = np.linalg.svd

    #fx, fy, cx, cy = 520, 520, 320, 240 
    fx, fy, cx, cy = 520, 520, image_width//2, image_height//2 

    # -----> Intrinsic Matrix of the Camera <----- #
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0,1]])
    K_inv = np.linalg.inv(K) 

    pose_sequence = [ np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ) ]
    max_frame = 900 #500

    





    # ----> Initialising Variables <----- # 
    Translation = np.zeros((3, 1))
    Rotation = np.eye(3)
    count = 0


    #cap = cv2.VideoCapture('/datasets/sample_videos/cambridge_office.MOV')
    cap = cv2.VideoCapture('/datasets/sample_videos/living_room.MOV')


    ret, key_frame_current = cap.read()  
    key_frame_current = process_resize(key_frame_current, image_width, image_height)
    current_frame = key_frame_current.copy()
    
    #initialize the visual odometry class
    func = VisualOdometry()


    while cap.isOpened():
        ret, key_frame_next = cap.read() 
        key_frame_next = process_resize(key_frame_next, image_width,image_height)
        cv2.imshow("frame ",key_frame_next)
        cv2.waitKey(1)

        next_frame = key_frame_next.copy()    


        if ret:

            point_correspondence_cf, point_correspondence_nf = func.extract_correspondences(current_frame,next_frame,use_superglue)
            R, t = func.find_pose_change(K,point_correspondence_cf,point_correspondence_nf,essential_estimation)
            Translation, Rotation, pose_sequence = func.accumulate_poses(Translation, Rotation, t, R, pose_sequence)

            count += 1
            print('# -----> Frame No:'+str(count),'<----- #', dt.now())
            if count==max_frame:
                break
        current_frame = next_frame

    func.visualize_trajectory(pose_sequence)



#to do 
# improve speed
# deal better with drifts
# incorporate visualization of sparse reprojected points