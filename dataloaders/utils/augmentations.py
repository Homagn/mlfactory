import numpy as np
import cv2
import math

#in many different datasets the sizes of images are largely inconsistent
#however ML models always use a fixed size input shape
#thus generally a simple resize is called to resize input
#however this might create unnatural/inconsistent aspect ratios for different objects in the image
#problem is aggravated in semantic segmentation type problems
#where model learning becomes un necessarily difficult
#therefore instead of straightforward resizing

#intelligently extract patches from the image if image size> model input size
#or intelligently upsize image and again extract patches in image size < model input size
#so that aspect ratios of individual objects are not distorted

def flip_images(image):
    img_flip_lr = cv2.flip(image, 1)
    return img_flip_lr

def calculate_stride(window_size, actual_size):
    #objective is to keep maximum length of stride
    #while fitting as many (window_size) number of patches
    wi = window_size
    w = actual_size
    if w%wi==0:
        w_stride = wi
    else:
        n_windows = math.ceil(w/wi)

        #print("w, wi",w,wi)
        w_stride = wi - int(((wi*n_windows)-w)/(n_windows-1)) - 1
    if w_stride==0:
        return 1
    return w_stride


def consistent_patches(random_size_image, model_input_size):
    patches = []
    #cv2.imshow("random size image ",random_size_image)
    #cv2.waitKey(0)

    w = random_size_image.shape[1]
    h = random_size_image.shape[0]
    #print("image shape height, width ",h,w)

    #this is the indexing of model input size
    wi = model_input_size[1]
    hi = model_input_size[0]

    #if image size is less than model input size
    if w<wi or h<hi:
        if w<=h:
            upscale = (1.0*wi)/w
        if h<w:
            upscale = (1.0*hi)/h
        random_size_image = cv2.resize(random_size_image,(int(w*upscale), int(h*upscale) ))

        #sometimes upscale is very small so resize doesnt happen
        if random_size_image.shape[1]<wi or random_size_image.shape[0]<hi:
            random_size_image = cv2.resize(random_size_image,(wi,hi))

        w = random_size_image.shape[1]
        h = random_size_image.shape[0]



        #print("rescaled height width ",h, w)
        #cv2.imshow("rescaled random size image ",random_size_image)
        #cv2.waitKey(0)



    w_stride = calculate_stride(wi,w)
    h_stride = calculate_stride(hi,h)



    
    
    for dh in range(0,h, h_stride):
        for dw in range(0,w, w_stride):
            #print("dh, dw ",dh,dw)
            patch = random_size_image[dh:dh+hi, dw:dw+wi]
            
            if patch.shape[0]!=wi or patch.shape[1]!=hi:
                #print("discarding invalid patch")
                continue
            #print("patch shape ",patch.shape)
            #cv2.imshow("patch ",patch)
            #cv2.waitKey(0)
            patches.append(patch)

    return patches

def augmented_windowed_patches(image, augmentations, size):
    patches = []

    #without any augmentations
    ps = consistent_patches( image, size)
    patches.extend(ps)
    
    #for each augmentation function
    for a in augmentations:
        ps = consistent_patches( a(image), size)
        patches.extend(ps)

    return patches


if __name__ == '__main__':
    im = cv2.imread("sample2.png")

    patches = augmented_windowed_patches(im, [flip_images], (256,256))
    print("final number of patches ",len(patches))
