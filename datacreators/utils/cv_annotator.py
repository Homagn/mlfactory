import numpy as np
import cv2
import math
import os
import json
import copy

#annotation functions
def create_supervisely_style_annotation_dict():
    d = {"description": "", "tags": [], "size": {"height":128, "width": 2048}, "objects":[]}
    return d

def load_annotation_dict(file):
    #file = annotation_folder+"ann/"+str(idx)+".png.json"
    if not os.path.exists(file):
        #print("This is first annotation for the image ")
        d = create_supervisely_style_annotation_dict()
    else:
        with open(file, 'r') as j:
            d = json.loads(j.read())
        #d = json.loads(file)
    return d

def populate_save_annotation_dict(file_name, d, object_box):
    #file = annotation_folder+"ann/"+str(idx)+".png.json"
    file = file_name
    a = {"geometryType": "rectangle", "points": {"exterior": [ [object_box[1], object_box[0]], [object_box[3], object_box[2]] ]  } }
    d["objects"].append(a)
    json_object = json.dumps(d, indent=4)
    with open(file, "w") as outfile:
        outfile.write(json_object)

def save_annotations(folder_name, image_file_name, idx, object_box, imread_func):

    file_name = folder_name+"ann/"+str(idx)+".png.json"
    d = load_annotation_dict(file_name)

    populate_save_annotation_dict(file_name, d,object_box)

    imfile = image_file_name
    imfilesave = folder_name+"img/"+str(idx)+".png"

    img = imread_func(imfile, 1)

    cv2.imwrite(imfilesave, img)
    print("saved image file name  ",imfile)


class bounding_box_annotator(object):
    def __init__(self, data, desired_size = (640,480) , normalize_bbox = True): #file name of the image to annotate
        
        if isinstance(data, str):
            print("file name has been supplied")
            self.image = cv2.imread(data)
        else:
            print("direct image information has been given ")
            self.image = data
        
        self.current_copy = "temp.png"
        self.normalize_bbox = normalize_bbox

        if desired_size==(-1,-1):
            self.desired_size = (self.image.shape[1],self.image.shape[0])
        else:
            self.desired_size = desired_size

        self.mouse_state = {'button':None, 'position':None, 'hover_position':None}

    def process_mouse_events(self, event,x,y,flags,param): #allows for continuous key press !
        # to check if left mouse  
        # button was clicked 
        if event == cv2.EVENT_LBUTTONDOWN: 
              
            # font for left click event 
            font = cv2.FONT_HERSHEY_TRIPLEX 
            LB = 'Left Button'
            #print("left button pressed at window 1 ",x,y)
            self.mouse_state['button'] = 'left'
            self.mouse_state['position'] = (x,y)

        elif event == cv2.EVENT_MOUSEMOVE:
            #print("mouse is in position ",x,y)
            self.mouse_state['hover_position'] = (x,y)
        else:
            #print("did not register mouse click ")
            self.mouse_state['button'] = None
            self.mouse_state['position'] = None

    def run(self):
        caption = "image"
        cv2.namedWindow(caption)
        cv2.setMouseCallback(caption, self.process_mouse_events)

        img1 = self.image #cv2.imread(self.sample)
        img1 = cv2.resize(img1, self.desired_size)

        imgb = self.image #cv2.imread(self.sample)
        imgb = cv2.resize(imgb, self.desired_size)
        cv2.imwrite(self.current_copy, imgb)
        #img1 = np.dstack((img1,img1,img1))
        #img2 = np.dstack((img2,img2,img2))
        src = []
        b = -1

        bboxes = []


        while(True):
            cv2.imshow(caption, img1)

            click_pts = []

            if self.mouse_state['position']!=None:
                pt1 = (self.mouse_state['position'][0],self.mouse_state['position'][1])
                click_pts = (self.mouse_state['position'][0],self.mouse_state['position'][1])
                self.mouse_state['button'] = None
                self.mouse_state['position'] = None 
                src.append(pt1)
                print("source points ",src)

            

            if click_pts!=[]:
                img1 = cv2.circle(img1, click_pts, 3, (255,255,255), -1)


            
            if len(src)%2!=0: #top left and bottom right clicks form two points for a bounding box
                hover_pts = (self.mouse_state['hover_position'][0],self.mouse_state['hover_position'][1])
                img1 = cv2.rectangle(imgb, src[-1], hover_pts, (255,255,255), 1)
                imgb = cv2.imread(self.current_copy)
                imgb = cv2.resize(imgb, self.desired_size)

            if len(src)%2==0 and len(src)!=0:
                cv2.imwrite(self.current_copy, img1)

                print(img1.shape, src[-2][0], src[-2][1])

                if self.normalize_bbox:
                    tl_x = src[-2][0]/img1.shape[1]
                    tl_y = src[-2][1]/img1.shape[0]

                    bl_x = src[-1][0]/img1.shape[1]
                    bl_y = src[-1][1]/img1.shape[0]
                else:
                    tl_x = src[-2][0]
                    tl_y = src[-2][1]

                    bl_x = src[-1][0]
                    bl_y = src[-1][1]


                bboxes.append([ int(tl_y), int(tl_x), int(tl_y+math.fabs(tl_y-bl_y)), int(tl_x+math.fabs(tl_x-bl_x))  ])


                print("current bounding boxes ",bboxes)
                src = []

                

            b = cv2.waitKey(1)
            if b == 110: #ord('n') keypress n
                print ("switching to next image ")
                # new_file = get_next_image(sample)

                # imgb = cv2.imread(new_file, cv2.IMREAD_GRAYSCALE)
                # cv2.imwrite(current_copy, imgb)

            if b == 99: #keypress c
                print ("clear points buffer ")
                #src = []

            if b == 27:
                print ("Ending annotation")
                break

        return bboxes


    def draw_assigned_boxes(self, image, boxes):
        img1 = image #cv2.imread(self.sample)
        img1 = cv2.resize(img1, self.desired_size)

        for box in boxes :
            #print("got box ",box)
            image = cv2.rectangle(img1, (box[1],box[0]), (box[3],box[2]), (255,255,255), 1)
        return image


    def adjust(self, boxes):
        caption = "adjusting_image"
        cv2.namedWindow(caption)
        cv2.setMouseCallback(caption, self.process_mouse_events)

        img1 = self.draw_assigned_boxes(self.image,boxes)
        img1 = cv2.resize(img1, self.desired_size)

        imgb = copy.copy(self.image)
        cv2.imwrite(self.current_copy, imgb)

        prev_boxes = copy.deepcopy(boxes)

        drag_mode = -1

        while(True):
            cv2.imshow(caption, img1)
            click_pts = []
            

            if self.mouse_state['position']!=None:
                pt1 = (self.mouse_state['position'][0],self.mouse_state['position'][1])
                click_pts = (self.mouse_state['position'][0],self.mouse_state['position'][1])

                distances = []
                for b in boxes:
                    dist = math.fabs(b[0]-click_pts[1])+math.fabs(b[1]-click_pts[0])
                    distances.append(dist)
                    #print("dist ",dist)
                
                box = boxes[np.argmin(distances)]
                #print("boxes ",boxes)
                #print("selected box number ",np.argmin(distances))

                print("click points ",click_pts)
                box_left_edge = math.fabs(box[1]-click_pts[0])
                box_top_edge = math.fabs(box[0]-click_pts[1])

                box_right_edge = math.fabs(box[3]-click_pts[0])
                box_bottom_edge = math.fabs(box[2]-click_pts[1])
                edge = np.argmin([box_left_edge,box_top_edge,box_right_edge,box_bottom_edge])


                self.mouse_state['button'] = None
                self.mouse_state['position'] = None 
                drag_mode = drag_mode*-1


            
            if self.mouse_state['position']==None and drag_mode>0:

                try:
                    hover_pts = (self.mouse_state['hover_position'][0],self.mouse_state['hover_position'][1])
                    
                    if edge==0:
                        #print("got left edge")
                        box[1] = hover_pts[0]
                    if edge==1:
                        #print("got top edge")
                        box[0] = hover_pts[1]
                    if edge==2:
                        #print("got right edge")
                        box[3] = hover_pts[0]
                    if edge==3:
                        #print("got bottom edge")
                        box[2] = hover_pts[1]

                    

                    
                    boxes[np.argmin(distances)] = box

                    img1 = self.draw_assigned_boxes(imgb,boxes)
                    imgb = cv2.imread(self.current_copy)
                    imgb = cv2.resize(imgb, self.desired_size)
                except:
                    #print("keep moving mouse ")
                    pass


            b = cv2.waitKey(1)


            if b == 27:
                print ("Ending annotation")
                #print("prev boxes ",prev_boxes)
                #print("current boxes ",boxes)
                adjustments = []
                for n in range(len(boxes)):
                    adj = list( (np.array(prev_boxes[n]) - np.array(boxes[n]))//2  )
                    adjustments.append(adj)
                
                break

        return boxes, adjustments










if __name__ == '__main__':
    ba = bounding_box_annotator("sample0.png", desired_size = (-1,-1), normalize_bbox = False)
    bboxes = ba.run()
    print("Final all bounding boxes ...")
    print(bboxes)

    ba = bounding_box_annotator("sample0.png", desired_size = (-1,-1), normalize_bbox = False)
    new_boxes, adjusts = ba.adjust(bboxes)
    print("got adjustments ",adjusts)



