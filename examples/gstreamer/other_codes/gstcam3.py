# source - https://github.com/basler/pypylon/issues/451
# doesnt work yet 



import sys;import traceback;import argparse;import typing as typ;import random;import time;from fractions import Fraction;import numpy as np

from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo, GLib, GstVideoSink
import gstreamer.utils as utils
#from pypylon import pylon
import cv2

VIDEO_FORMAT = "I420" #BGR #GRAY8 #I420
WIDTH, HEIGHT = 640, 480 #1440
FPS = Fraction(25)
GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(VIDEO_FORMAT)

def fraction_to_str(fraction: Fraction) -> str:
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)

FPS_STR = fraction_to_str(FPS)
DEFAULT_CAPS = "video/x-raw,format={VIDEO_FORMAT},width={WIDTH},height={HEIGHT},framerate={FPS_STR}".format(**locals())
#x-rtp-stream,encoding-name=JPEG


'''
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
'''

camera = cv2.VideoCapture(0)

'''
DEFAULT_PIPELINE = utils.to_gst_string([
    "appsrc emit-signals=True is-live=True caps={DEFAULT_CAPS}".format(**locals()),
    "queue",
    "videoscale",
    "videoconvert",
    "x264enc tune=zerolatency bitrate=500 speed-preset=superfast",
    "rtph264pay", 
    #"autovideosink"
    "udpsink host=192.168.1.116 port=5200"    
    
    
])
'''

DEFAULT_PIPELINE = utils.to_gst_string([
    "appsrc emit-signals=True is-live=True caps={DEFAULT_CAPS}".format(**locals()),
    "queue",
    "videoscale",
    "videoconvert",
    "x264enc tune=zerolatency bitrate=500 speed-preset=superfast",
    "rtph264pay", 
    #"autovideosink",
    "udpsink host=255.255.255.255 port=5000"    
    
    
])

command = DEFAULT_PIPELINE
NUM_BUFFERS=10000
GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(VIDEO_FORMAT)
CHANNELS = utils.get_num_channels(GST_VIDEO_FORMAT)
DTYPE = utils.get_np_dtype(GST_VIDEO_FORMAT)
CAPS = DEFAULT_CAPS

with GstContext():  # create GstContext (hides MainLoop)
    pipeline = GstPipeline(command)
    def on_pipeline_init(self):
        """Setup AppSrc element"""
        appsrc = self.get_by_cls(GstApp.AppSrc)[0]  # get AppSrc        
        appsrc.set_property("format", Gst.Format.TIME) # instructs appsrc that we will be dealing with timed buffer
        appsrc.set_property("block", True)# instructs appsrc to block pushing buffers until ones in queue are preprocessed # allows to avoid huge queue internal queue size in appsrc        
        appsrc.set_caps(Gst.Caps.from_string(CAPS)) # set input format (caps)    
    pipeline._on_pipeline_init = on_pipeline_init.__get__(pipeline) # override on_pipeline_init to set specific properties before launching pipeline
    try:
        pipeline.startup()
        appsrc = pipeline.get_by_cls(GstApp.AppSrc)[0]  # GstApp.AppSrc
        pts = 0  # buffers presentation timestamp
        duration = 10**9 / (FPS.numerator / FPS.denominator)  # frame duration
        for nb in range(NUM_BUFFERS):
            #grabResult = camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException) #5000
            if camera.isOpened():
                ret, grabResult = camera.read()
                cv2.imshow("image ",grabResult)
                cv2.waitKey(1)
            #if grabResult.GrabSucceeded():
            if ret:
                print("got frame ",nb)
                #image = converter.Convert(grabResult)
                #array = image.GetArray()
                #array = grabResult.GetArray()
                array = np.asarray(grabResult).astype(np.uint8)
                #print(array.size)
                # create random np.ndarray
                ##array = np.random.randint(low=0, high=255,
                ##                        size=(HEIGHT, WIDTH, CHANNELS), dtype=DTYPE)            
            gst_buffer = utils.ndarray_to_gst_buffer(array)# convert np.ndarray to Gst.Buffer            
            pts += duration  # # set pts and duration to be able to record video, calculate fps #Increase pts by duration
            gst_buffer.pts = pts
            gst_buffer.duration = duration            
            retval = appsrc.emit("push-buffer", gst_buffer) # emit <push-buffer> event with Gst.Buffer       
            #print("retval ",retval) 
        appsrc.emit("end-of-stream")# emit <end-of-stream> event

        while not pipeline.is_done:
            time.sleep(.1)
    except Exception as e:
        print("Error: ", e)
    finally:
        pipeline.shutdown()