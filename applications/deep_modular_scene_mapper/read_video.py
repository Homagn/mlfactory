import cv2
import numpy as np
import os
import sys
import shutil

def extract_video(video_loc = '/datasets/sample_videos/cambridge_office.MOV', extract_loc = '/datasets/sample_videos/extracted/'):

  if os.path.exists(extract_loc):
    inp = input("save path location already exists, delete and create fresh ? (y/n) ")
    if inp=='y':
        print("deleting save loc and creating fresh ")
        shutil.rmtree(extract_loc)
        print("deleted")
        os.makedirs(extract_loc)

  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture(video_loc)
   
  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
   
  # Read until video is completed
  n = 0
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    
      new_height = 480 if frame.shape[0] > 480 else frame.shape[0]
      new_height -= (new_height % 32)
      new_width = int(new_height * frame.shape[1] / frame.shape[0])
      diff = new_width % 32
      new_width = new_width - diff if diff < 16 else new_width + 32 - diff
      new_size = (new_width, new_height)
      #image = image.resize(new_size)
      frame = cv2.resize(frame, new_size)

      # Display the resulting frame
      cv2.imshow(str(n),frame)
      cv2.imwrite(extract_loc+str(n)+".jpg",frame)
      n+=1
   
      # Press Q on keyboard to  exit
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break
   
    # Break the loop
    else: 
      break
   
  # When everything done, release the video capture object
  cap.release()
   
  # Closes all the frames
  cv2.destroyAllWindows()


if __name__ == '__main__':
  extract_video()

