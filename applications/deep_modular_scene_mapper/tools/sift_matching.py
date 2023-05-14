import cv2
import sys
import os.path
import numpy as np
#pip install scikit-image
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import time


def prepare_images(filename1, filename2):
    img1 = cv2.imread(filename1)          # queryImage
    img2 = cv2.imread(filename2)          # trainImage

    width = img1.shape[1]

    img1 = img1[:,:width//2]
    img2 = img2[:,:width//2]

    res_shape = (256,256)

    img1 = cv2.resize(img1,res_shape)
    img2 = cv2.resize(img2,res_shape)

    #img1 = np.flipud(img1)
    #img2 = np.flipud(img2)

    return img1, img2




def drawMatches(img1, kp1, img2, kp2, matches, limit=-1):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1])
    out[:rows2,cols1:] = np.dstack([img2])
    for mat in matches[:limit]:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0, 1), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0, 1), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0, 1), 1)

    return out

def compare(filename1, filename2):

    img1,img2 = prepare_images(filename1,filename2)


    # Initiate SIFT detector
    #sift = cv2.SIFT_create()
    sift = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)

    matches = sorted(matches, key=lambda val: val.distance)

    #print("matches in compare ",matches)

    img3 = drawMatches(img1,kp1,img2,kp2,matches[:25])

    # Show the image
    cv2.imshow('Matched Features', img3)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')


def compare_extra(filename1,filename2): #includes ration test and ransac test


    img1,img2 = prepare_images(filename1,filename2)
    #surf = cv2.SURF_create(100)
    surf = cv2.SIFT_create()
    #surf = cv2.ORB_create()

    # surf = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann = cv2.BFMatcher()

    matches = flann.knnMatch(des1,des2,k=2)

    #print("matches in compare_extra ",matches)

    # Lowe's Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

    print("sift matching got number of matches  ",src_pts.shape[0])
    # Ransac
    model, inliers = ransac(
            (src_pts, dst_pts),
            ProjectiveTransform, min_samples=4,
            residual_threshold=8, max_trials=100
        )

    

    n_inliers = np.sum(inliers)

    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
    
    #image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
    image3 = drawMatches(img1,inlier_keypoints_left,img2,inlier_keypoints_right,placeholder_matches)


    cv2.imshow('Matches', image3)
    cv2.waitKey(0)

    src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
    dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

    #print("src_pts ",src_pts)
    #print("dst_pts ",dst_pts)

    return src_pts, dst_pts


if __name__ == '__main__':
    
    #compare("data/test/190.png", "data/test/200.png")
    #compare_extra("data/test/190.png", "data/test/200.png")

    id1 = 160
    id2 = 170

    compare("data/table/"+str(id1)+".png", "data/table/"+str(id2)+".png")


    s,d = compare_extra("data/table/"+str(id1)+".png", "data/table/"+str(id2)+".png")

    #print("source points ",s)
    #print("dest points ",d)