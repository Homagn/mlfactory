import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output




import numpy as np
import cv2
import math
import os
import json
import copy
import sys




class poly_annotator(object):
    def __init__(self, data, desired_size = (640,480) , normalize_bbox = True, start_index = 0, workspace = "trial", max_annotations_order = 3): #file name of the image to annotate
        self.start_index = start_index
        self.data = data
        self.max_annotations_order = max_annotations_order
        print("file name has been supplied")
        self.image = cv2.imread(data[start_index])
        self.workspace = workspace

       
        self.current_copy = "temp.png"
        self.normalize_bbox = normalize_bbox

        if desired_size==(-1,-1):
            self.desired_size = (self.image.shape[1],self.image.shape[0])
        else:
            self.desired_size = desired_size

        self.mouse_state = {'button':None, 'position':None, 'hover_position':None}


    def onclick(self,event):
        ix, iy = event.xdata, event.ydata
        #print("clicked ",ix, iy)
        self.process_mouse_events("click",ix,iy)

    def hover(self,event):
        ix, iy = event.xdata, event.ydata
        #print("hover ",ix, iy)
        self.process_mouse_events("hover",ix,iy)
    
    def on_key_press(self, event):
        #print('press', event.key)
        sys.stdout.flush()
        if event.key == 'x':
            self.process_mouse_events("key",-1,-1)

    def process_mouse_events(self, flag,x,y): #allows for continuous key press !
        # to check if left mouse  
        # button was clicked
        if flag == "click":
            self.mouse_state['button'] = 'left'
            try:
              self.mouse_state['position'] = (int(x),int(y))
              #print("saved click")
              self.img1 = copy.copy(self.imgt)
              self.imgb = copy.copy(self.imgt)
              self.im.set_data(self.img1)
              
            except:
              pass

        elif flag == "hover":
            #print("mouse is in position ",x,y)
            try:
                self.mouse_state['hover_position'] = (int(x),int(y))
                click_pts = []

                if self.mouse_state['position']!=None:
                    pt1 = (self.mouse_state['position'][0],self.mouse_state['position'][1])
                    click_pts = (self.mouse_state['position'][0],self.mouse_state['position'][1])
                    self.mouse_state['button'] = None
                    self.mouse_state['position'] = None
                    self.src.append(pt1)
                    



                if click_pts!=[]:
                    self.img1 = cv2.circle(self.img1, click_pts, 3, (255,255,255), -1)
                    #self.imgt = copy.copy(self.img1)


                hover_pts = (self.mouse_state['hover_position'][0],self.mouse_state['hover_position'][1])
                if len(self.src)%2!=0: #top left and bottom right clicks form two points for a bounding box
                  
                  self.imgt = cv2.line(self.img1, self.src[-1], hover_pts, (255,255,255), 1)
                  #self.imgb = copy.copy(self.imstore)
                  #self.imgb = cv2.resize(self.imgb, self.desired_size)

                if len(self.src)%2==0 and len(self.src)!=0:
                  self.imgt = cv2.line(self.img1,hover_pts,self.src[-1], (255,255,255), 1)
                    
                self.im.set_data(self.imgt)
                self.img1 = copy.copy(self.imgb)
              
            
            


            except:
                pass

        elif flag=="key":
            if not os.path.exists(self.workspace):
              os.mkdir(self.workspace)
              os.mkdir(self.workspace+"/Images")
              os.mkdir(self.workspace+"/Masks")
            
            self.mouse_state['button'] = None
            self.mouse_state['position'] = None
            #self.imgb = copy.copy(self.imstore)
            print("source points ",self.src)
            hull = cv2.convexHull(np.array(self.src), False)
            drawing = np.zeros((self.img1.shape[0], self.img1.shape[1], 3), np.uint8)
            #cv2.drawContours(drawing, [hull], 0, (255, 255, 255), 1, 8)
            cv2.fillPoly(drawing, pts =[hull], color=(255,255,255))
            str_code = str(self.start_index)
            strc = ''
            for j in range(self.max_annotations_order-len(str_code)):
              strc = strc+'0'
            strc = strc+str_code

            #save the image and the masks for segmentation training
            cv2.imwrite(self.workspace+"/Masks/"+strc+"_label.PNG",drawing)
            cv2.imwrite(self.workspace+"/Images/"+strc+".jpg",self.imstore)


            self.src = []
            self.start_index+=1
            if self.start_index<len(self.data):

                self.imstore = cv2.imread(self.data[self.start_index])
                self.imstore = cv2.resize(self.imstore, self.desired_size)
                self.img1 = copy.copy(self.imstore)
                self.imgb = copy.copy(self.imstore)
                self.imgt = copy.copy(self.imstore)
            else:
                print("annotated all supplied image names ")
                plt.close()
            


        else:
            pass

        #self.imstore = copy.copy(self.img1)

    def run(self):
        caption = "image"
        #cv2.namedWindow(caption)
        #cv2.setMouseCallback(caption, self.process_mouse_events)
        self.done = False
        img1 = self.image #cv2.imread(self.sample)
        self.img1 = cv2.resize(img1, self.desired_size)
        self.imgt = copy.copy(self.img1)

        self.imgb = copy.copy(self.img1)
        #self.imgb = cv2.resize(self.imgb, self.desired_size)
        self.imstore = copy.copy(self.imgb)

        cv2.imwrite(self.current_copy, self.imgb)
        #img1 = np.dstack((img1,img1,img1))
        #img2 = np.dstack((img2,img2,img2))
        self.src = []
        b = -1

        self.bboxes = []
        self.forming=-1

        #clear_output()
        fig, ax = plt.subplots()
        ax.set_axis_off()
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid2 = fig.canvas.mpl_connect("motion_notify_event", self.hover)
        cid3 = fig.canvas.mpl_connect("key_press_event", self.on_key_press)
      
        self.im = ax.imshow(self.img1)
        #plt.show()
        import sys
        sys.stdout.flush()

