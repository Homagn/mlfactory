import os
import copy

import cv2
import math
import json

class jsonwriter(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = []
    def cwrite(self,data):
        if os.path.exists(self.filename):
            #print("json file has been written once ")
            with open(self.filename, 'r') as openfile:
                # Reading from json file
                json_object = json.load(openfile)
                self.data = json_object
        self.data.extend(data)

        # Serializing json
        json_object = json.dumps(self.data, indent=4)
         
        # Writing to sample.json
        with open(self.filename, "w") as outfile:
            outfile.write(json_object)



#usage
if __name__ == '__main__':
    js = jsonwriter("samplelabels.json")
    data = [{"id": 'sample.png', "dataattribute1": (0,1,2), "dataattribute2": (1.2,3.5,6.7) }]

    print("writing data ",data)
    js.cwrite(data)