#Installation
'''

python3 -m pip install pyautogui

sudo apt-get install scrot

sudo apt-get install python3-tk

sudo apt-get install python3-dev

pip install python-xlib

(for mss.tool fast screengrab )
pip install mss

'''

import pyautogui
import cv2
import numpy as np
from cv_annotator import bounding_box_annotator
import time

#pip install keyboard
import keyboard  # using module keyboard

#pip install pynput
from pynput.keyboard import Key, Listener
import threading
from jsonlabels import jsonwriter
from datetime import datetime as dt

import mss.tools #for fast screenshots


folder = "/datasets/behavior_cloning/maze_game/game15/"
idx = 0
js = jsonwriter(folder+"samplelabels.json")
current = set()

mode = 'record' # 'find_region', 'record'


def find_interesting_region():
    im2 = pyautogui.screenshot('full_screenshot.png') #take full screen screenshot

    #invoke bounding box annotator so that user can draw a box to query coordinates of interest area
    ba = bounding_box_annotator("full_screenshot.png", desired_size = (-1,-1), normalize_bbox = False)
    bboxes = ba.run()

    left = bboxes[0][1]
    top = bboxes[0][0]
    width = bboxes[0][3] - bboxes[0][1]
    height = bboxes[0][2] - bboxes[0][0]
    print("left, top, width, height ",left, top, width, height)
    # Take note of the four printed values above and use it as below commented code

    #im = pyautogui.screenshot(region=(left, top, width, height))



def record_screen(folder, idx, save_img_size = (256,256)):
    #im = pyautogui.screenshot(region=(823, 153, 997, 464)) #takes around 0.15 s
    #cv2.imwrite(folder+"/"+str(idx)+".png", np.array(im))

    with mss.mss() as sct:
        # The screen part to capture
        monitor = {"top": 191, "left": 373, "width": 1125, "height": 611}
        #output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)
        output = folder+"/"+str(idx)+".png"

        # Grab the data
        sct_img = np.array(sct.grab(monitor)) #takes around 0.03 s
        #print("sct image size ",sct_img.size)
        # Save to the picture file
        #mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

        cv_img = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR)
        cv_img = cv2.resize(cv_img, save_img_size)

        cv2.imwrite(output, cv_img)
        #mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        #print(output)




def capture_screen():
    global idx
    global folder
    global current
    global js
    while True:
        record_screen(folder,idx)
        idx+=1
        kp = [str(k) for k in current]
        data = [{"id": str(idx)+'.png', "keys_pressed": kp, "time": str(dt.now()) }]
        js.cwrite(data)


class rec_keys(object):
    def __init__(self):
        self.current = set()

    def on_press(self, key):
        global current
        current.add(key)
        for k in self.current:
            print('{0} pressed'.format(k), end='')
        print('')

    def on_release(self, key):
        global current
        try:
            current.remove(key)
            print('{0} release'.format(key))
        except KeyError:
            pass
        if key == Key.esc:
            # Stop listener
            return False
    def start(self):
        # Collect events until released
        with Listener(on_press=self.on_press,on_release=self.on_release) as listener:
            listener.join()


if __name__ == '__main__':
    if mode == 'find_region':
        find_interesting_region()

    if mode == 'record':
        rk = rec_keys()

        thread = threading.Thread(target=capture_screen, args=())
        thread.start()

        rk.start()