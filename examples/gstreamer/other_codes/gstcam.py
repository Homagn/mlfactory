import time
import cv2

# Cam properties
fps = 30.
frame_width = 640
frame_height = 480
# Create capture
cap = cv2.VideoCapture(0)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, fps)


fourcc = cv2.VideoWriter_fourcc(*'MJPG')


'''
tests
gst-launch-1.0 -v videotestsrc ! decodebin ! x264enc bitrate=5000 ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5000

apt-get install tcpdump

tcpdump -i any port 5000


'''
# Define the gstreamer sink

#gst_str_rtp = "appsrc ! videoconvert ! 264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=5000"

#gst_str_rtp = "appsrc ! videoconvert ! videoscale ! video/x-raw,format=I420,width=640,height=480,framerate=5/1 !  videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=5000"

gst_str_rtp = "appsrc ! videoconvert ! x264enc noise-reduction=10000 tune=zerolatency byte-stream=true threads=4 ! h264parse ! mpegtsmux ! rtpmp2tpay ! udpsink host=127.0.0.1 port=5000"


gst_str_rtp = 'appsrc  ! videoconvert ! h264parse ! rtph264pay config-interval=1 pt=96 ! gdppay ! tcpserversink host=127.0.0.1 port=5000 '

gst_str_rtp = 'appsrc ! queue ! videoconvert ! video/x-raw ! h264enc ! video/x-h264 ! h264parse ! rtph264pay ! udpsink host=127.0.0.1 port=5000 sync=false'

# Check if cap is open
if cap.isOpened() is not True:
    print ("Cannot open camera. Exiting.")
    quit()

# Create videowriter as a SHM sink
#out = cv2.VideoWriter(gst_str_rtp, 0, fps, (frame_width, frame_height), True)
out = cv2.VideoWriter(gst_str_rtp, cv2.CAP_GSTREAMER, 0, fps, (frame_width, frame_height))


# Loop it
while True:
    # Get the frame
    ret, frame = cap.read()
    #cv2.imshow("frame ",frame)
    #cv2.waitKey(1)
    # Check
    if ret is True:
        # Flip frame
        frame = cv2.flip(frame, 1)
        # Write to SHM
        out.write(frame)
    else:
        print ("Camera error.")
        time.sleep(10)

cap.release()