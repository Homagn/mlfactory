'''
About:
take an image selects 4 points in the image
slightly shifts those 4 points in the image
using the prev and new 4 points, calculates a homography matrix
generates a random perspective distortion of the image
'''



import numpy as np
import cv2

desired_size = (512,512)
patch_size = (350,350)
warp_factor = 30
min_crop_border = 40 #should be greater than warp_factor

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])    
    return frame


def remove_black_background(frame):
    #crop top
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            try:
                if frame[i-1,j]==0 and frame[i,j-1]==0:
                    print("found hanging top left corner at ",i,j)
            except:
                pass
    return frame


def uniform_zoom(mat):
    import copy
    mat2 = copy.copy(mat)
    val = 20

    mat2[0,0] +=20
    mat2[0,1] +=20

    mat2[1,0] +=20
    mat2[1,1] +=20

    mat2[2,0] +=20
    mat2[2,1] +=20

    mat2[3,0] -=20
    mat2[3,1] -=20

    return mat2


def random_misalign(img, cx, cy):
    print(crop_y1,crop_x1)

    img = img+np.ones_like(img)
    

    src = np.float32([[cx,cy],[cx+patch_size[0],cy],[cx,cy+patch_size[1]],[cx+patch_size[0],cy+patch_size[1]]])
    
    #basically adding random numbers to each element of the src matrix
    dst = np.float32([ [cx + int(np.random.rand()*warp_factor) , cy + int(np.random.rand()*warp_factor) ],
            [cx + patch_size[0] + int(np.random.rand()*warp_factor), cy + int(np.random.rand()*warp_factor) ],
            [cx + int(np.random.rand()*warp_factor) , cy + patch_size[1] + int(np.random.rand()*warp_factor) ],
            [cx + patch_size[0] + int(np.random.rand()*warp_factor) , cy+ patch_size[1] +int(np.random.rand()*warp_factor) ] ])

    #dst = uniform_zoom(src)

    
    #M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    M = cv2.getPerspectiveTransform(src, dst)
    print("perspective matrix ",M)

    M,_ = cv2.findHomography(src, dst)
    print("perspective matrix ",M)

    camera_matrix = np.array([[118.62248444,  0.0,         165.91476167],
                             [  0.0 ,         99.79947857, 175.91749674],
                             [  0.0 ,          0.0,           1.0        ]])

    num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(M, camera_matrix)
    for n in range(num):
        print("decomposed rotations ",Rs[n])
        print("decomposed translations ",Ts[n])


    print("corner difference matrix ",src-dst)
    warped = cv2.warpPerspective(img, M, desired_size ,flags=cv2.INTER_LINEAR)

    
    

    cv2.imshow('original ',img)
    cv2.imshow('warped ',warped)

    wcrop = warped [ cx: cx+patch_size[0], cy: cy+patch_size[1] ]
    cv2.imshow('warped cropped',wcrop)


    '''
    homography_inverse = np.linalg.inv(M)
    crop_corrected = cv2.warpPerspective(img[ cx: cx+patch_size[0], cy: cy+patch_size[1] ] , homography_inverse, patch_size ,flags=cv2.INTER_LINEAR)
    cv2.imshow('crop_corrected',crop_corrected)
    '''

    cv2.waitKey(0)


    trimm = trim(warped)
    #fit = remove_black_background(trimm[10:-10,10:-10])
    #fit_warped = remove_black_background(warped)
    #cv2.imshow("fit warped ",fit_warped)
    
    #cv2.imshow("fit ",fit)
    cv2.waitKey(0)


    return M




img1 = cv2.imread("extracted/100.png")          # queryImage
img2 = cv2.imread("extracted/120.png")          # trainImage

width = img1.shape[1]

img1 = img1[:,:width//2]
img2 = img2[:,:width//2]

img1 = cv2.resize(img1,(512,512))
img2 = cv2.resize(img2,(512,512))


crop_x1 = min_crop_border + int(np.random.rand()*(desired_size[0]-patch_size[0]-min_crop_border))
crop_y1 = min_crop_border + int(np.random.rand()*(desired_size[1]-patch_size[1]-min_crop_border))

print(crop_y1,crop_x1)

crop1 = img1[ crop_x1:crop_x1+patch_size[0], crop_y1:crop_y1+patch_size[1] ]

cv2.imshow('original cropped',crop1)
#cv2.waitKey(0)

random_misalign(img1,crop_x1,crop_y1)


