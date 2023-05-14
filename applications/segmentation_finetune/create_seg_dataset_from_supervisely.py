import cv2
import numpy as np
import os
import sys
import shutil


#if an image is huge and contains multiple segmentations across the width then we can tile it and generate multiple data from one image
class repeated_pattern_patches(object):
    def __init__(self, imsize = [2048,128], num_horizontal_tiles = 12):
        self.w = imsize[0]
        self.h = imsize[1]
        self.nht = num_horizontal_tiles

    def patch(self,img):
        r = self.w//self.nht
        patches = []
        for i in range(self.nht):
            p = img[:,i*r:(i+1)*r]
            patches.append(p)
        return patches


def create_seg_dataset(image_mask_location = "/datasets/lidar_post_detection/pillar_seg/extracted_lidar_data/",
                        save_data_location = "/datasets/small_segmentation/lidar_pillars/",
                        start = 210,
                        end = 390,
                        skip = 10,
                        patcher = None):
    
    if os.path.exists(save_data_location):
        inp = input("save path location already exists, delete and create fresh ? (y/n) ")
        if inp=='y':
            print("deleting save loc and creating fresh ")
            shutil.rmtree(save_data_location)
            print("deleted")
            os.makedirs(save_data_location+"Masks")
            os.makedirs(save_data_location+"Images")
    if not os.path.exists(save_data_location):
        print("data directory does not exist creating one")

        os.makedirs(save_data_location+"Masks")
        os.makedirs(save_data_location+"Images")

    
    count = 1
    strc = "001"

    for i in range(start, end, skip):
        

        a = cv2.imread(image_mask_location+"masks_machine/"+str(i)+".png")
        b = cv2.imread(image_mask_location+"img/"+str(i)+".png")

        a = a*255.0
        if patcher!=None:
            a_patches = patcher.patch(a)
            b_patches = patcher.patch(b)
        else:
            a_patches = a
            b_patches = b
        

        for k in range(len(a_patches)):
            ai = cv2.resize(a_patches[k], (480,320))
            cv2.imshow("mask ",ai*255.0)
            cv2.waitKey(1)

                
            bi = cv2.resize(b_patches[k], (480,320))
            cv2.imshow("image ",bi)
            cv2.waitKey(1)

            if np.mean(bi)>10.0: #majority of image patch does not contain any segmentation mask
                print(np.max(bi),np.min(bi),np.mean(bi))
                cv2.imwrite(save_data_location+"Masks/"+strc+"_label.PNG", ai)
                cv2.imwrite(save_data_location+"Images/"+strc+".jpg", bi)

                count+=1



                str_code = str(count)
                strc = ""
                for j in range( 3-len(str_code) ) :
                    strc = strc+"0"
                strc = strc+str_code
                print("got strc ",strc)



if __name__ == '__main__':
    create_seg_dataset(patcher = repeated_pattern_patches())