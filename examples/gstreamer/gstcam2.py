'''

this code source - https://gist.github.com/robobe/525c1155f47e4422127c715332d29bb5#file-cv2gst_pipe-L24

install gstreamer
apt-get update
apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

installing gst-python
(first make sure to upgrade pip)
pip install --upgrade pip
pip install --upgrade gstreamer-python@git+https://github.com/jackersson/gstreamer-python.git#egg=gstreamer-python


Running this code will publish opencv images captured from the camera frame by frame as a gstreamer video


to see the output of the rtp stream that is being published on the localhost, type this in the terminal:
gst-launch-1.0 -v udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! autovideosink

'''



#!/usr/bin/env python3
import sys
import os
import cv2
import gi
import signal
import threading
import time as t

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib

#GObject.threads_init() #change
Gst.init(None)
class video():
    def __init__(self):
        self.number_frames = 0
        self.fps = 40
        self.duration = 1 / self.fps * Gst.SECOND

        #jpec encoding dont use it John says and it also doesnt work
        '''
        self.pipe = "appsrc name=source is-live=true block=true format=GST_FORMAT_TIME " \
            " caps=video/x-raw,format=BGR,width=640,height=480,framerate={}/1 " \
            "! videoconvert ! video/x-raw,format=I420 " \
            "!  jpegenc " \
            "! rtpjpegpay " \
            "! udpsink host=127.0.0.1 port=5000".format(self.fps)
        '''

        #x264 which is needed by dm1000
        '''
        self.pipe = "appsrc name=source is-live=true block=true format=GST_FORMAT_TIME " \
            " caps=video/x-raw,format=BGR,width=640,height=480,framerate={}/1 " \
            "! videoconvert ! video/x-raw,format=I420 " \
            "!  x264enc " \
            "! rtph264pay " \
            "! udpsink host=127.0.0.1 port=5000".format(self.fps)
        '''

        '''
        #suggested by John to change the host to 255.255.255.255 (sends to every device in the network)
        #may not work if port is changed to 5555
        self.pipe = "appsrc name=source is-live=true block=true format=GST_FORMAT_TIME " \
            " caps=video/x-raw,format=BGR,width=640,height=480,framerate={}/1 " \
            "! videoconvert ! video/x-raw,format=I420 " \
            "!  x264enc " \
            "! rtph264pay " \
            "! udpsink host=255.255.255.255 port=5000".format(self.fps)
        '''
        

        
        self.pipe = "appsrc name=source is-live=true block=true format=GST_FORMAT_TIME " \
            " caps=video/x-raw,format=BGR,width=640,height=480,framerate={}/1 " \
            "! videoconvert ! video/x-raw,format=I420 " \
            "!  x264enc tune=zerolatency " \
            "! rtph264pay " \
            "! udpsink host=255.255.255.255 port=5000".format(self.fps)
        

        
        
        

        '''
        #for testing purposes (no need of receiver pipeline)
        self.pipe = "appsrc name=source is-live=true block=true format=GST_FORMAT_TIME " \
            " caps=video/x-raw,format=BGR,width=640,height=480,framerate={}/1 " \
            "! videoconvert ! video/x-raw,format=NV12 " \
            "! x264enc " \
            "! queue " \
            "! decodebin " \
            "! autovideosink sync=false".format(self.fps)
        '''
        




        self.cap = cv2.VideoCapture(0)

        self.pipeline = Gst.parse_launch(self.pipe)
        self.loop = None
        appsrc=self.pipeline.get_by_name('source')
        appsrc.connect('need-data', self.on_need_data)
        
        

    def run(self):
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.PLAYING)
        
        self.loop = GLib.MainLoop() #GObject.MainLoop() #change
        self.loop.run()

    def on_need_data(self, src, lenght):
       if self.cap.isOpened():
            ret, frame = self.cap.read()
            #cv2.imwrite("static_frame.png",frame)
            if ret:
                data = frame.tobytes() #frame.tostring() #change
                
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit('push-buffer', buf)
                print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                                                                                       self.duration,
                                                                                   self.duration / Gst.SECOND))
                print(retval)                                                                                   
                if retval != Gst.FlowReturn.OK:
                    print(retval)
                return True

        

if __name__ == "__main__":
    """
    gst-launch-1.0 -v udpsrc port=1234 \
    ! application/x-rtp, encoding-name=JPEG,payload=26 \
    ! rtpjpegdepay \
    ! jpegdec \
    ! autovideosink
    """
    v = video()
    v.run()