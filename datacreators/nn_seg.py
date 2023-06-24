

import torch
import cv2
import pandas as pd
import numpy as np



class segmenter(object):
    def __init__(self, loadpath = 'exp/weights.pt', viz = True):
        # Load the trained model 
        self.model = torch.load(loadpath)
        # Set the model to evaluate mode
        self.model.eval()
        print("loaded segmentation model ")
        self.border = 60
        self.viz = viz
        self.model_input_shape = (480,320)

    def run_predictor(self, imfile):
        ino = 34 #48 #44
        # Read  a sample image and mask from the data-set
        self.base_image = cv2.imread(imfile)
        img = cv2.resize(self.base_image, self.model_input_shape)
        self.base_image = img

        #img = cv2.imread(imfile).transpose(2,0,1).reshape(1,3,320,480)
        img = img.transpose(2,0,1).reshape(1,3,320,480)

        #mask = cv2.imread(f'./food_packet/Masks/{ino:03d}_label.PNG')
        #print("mask shape ",mask.shape)

        with torch.no_grad():
            a = self.model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
            print("model pred success ")

        pred_mask = np.zeros((320,480))
        pred_mask[a['out'].cpu().detach().numpy()[0][0]>0.2] = 1.0
        #pred_mask[pred_mask<255.0] = 50
        if self.viz:
            cv2.imshow("pred_mask ",pred_mask)
            cv2.waitKey(0)

        self.pred_mask = pred_mask

    def extract_roi(self, savefilename = "../pose_pred_in.png"):
        mask = np.array(self.pred_mask).astype(np.uint8)
        masked = cv2.bitwise_and(self.base_image, self.base_image, mask=mask)
        masked[np.all(masked == (0, 0, 0), axis=-1)] = (127,127,127)

        if self.viz:
            cv2.imshow("Mask Applied to Image", masked)
            cv2.waitKey(0)



        contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # take the first contour
        cnt = contours[0]

        # compute the bounding rectangle of the contour
        x,y,w,h = cv2.boundingRect(cnt)

        # draw contour
        #masked = cv2.drawContours(masked,[cnt],0,(0,255,255),2)

        # draw the bounding rectangle
        #masked = cv2.rectangle(masked,(x,y),(x+w,y+h),(0,255,0),2)

        extra_border = self.border

        masked_area = masked[y:y+h,x:x+w]
        pose_pred_in_image = np.zeros( ( masked_area.shape[0]+extra_border, masked_area.shape[1]+extra_border , 3) ).astype(np.uint8)
        pose_pred_in_image[np.all(pose_pred_in_image == (0, 0, 0), axis=-1)] = (127,127,127)
        pose_pred_in_image.astype(np.uint8)
        pose_pred_in_image[extra_border//2:-extra_border//2,extra_border//2:-extra_border//2, :] = masked_area

        pose_pred_in_image = cv2.resize(pose_pred_in_image,(224,224))
        cv2.imwrite(savefilename,pose_pred_in_image)

        # display the image with bounding rectangle drawn on it
        if self.viz:
            cv2.imshow("Bounding Rectangle", pose_pred_in_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return x,y,w,h #= x,y,w,h


    def extract_boxes(self, resize_boxes = (170,128) ):
        mask = np.array(self.pred_mask).astype(np.uint8)


        

        mask = cv2.resize(mask,resize_boxes)
        res_img = cv2.resize(self.base_image,resize_boxes)

        contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for cont in contours:
            # take the first contour
            cnt = cont
            # compute the bounding rectangle of the contour
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append([x,y,w,h])
            mask = cv2.rectangle(res_img,(x,y),(x+w,y+h),(0,255,0),2)
        
        if self.viz:
            cv2.imshow("rectangle ",mask)
            cv2.waitKey(0)

        return boxes












if __name__ == '__main__':
    ino = 33 #48 #44
    s = segmenter()
    #s.run_predictor(f'./food_packet/Images/{ino:03d}.jpg')
    s.run_predictor(f'/datasets/small_segmentation/lidar_pillars/Images/'+"033"+'.jpg')
    boxes = s.extract_boxes()
    print("got bounding box annotations ",boxes)

    #x,y,w,h = s.extract_roi(savefilename = "output.png")



