
gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)224, height=(int)224, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)224, height=(int)224, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
#gst_str = ("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, format=(string)YUY2,framerate=30/1 ! videoconvert ! video/x-raw,width=640,height=480,format=BGR ! appsink")


import cv2
import numpy as np
import datetime
import time
import os
from uuid import uuid1

coord_x = 0
coord_y = 0


def mouse_callback(event, x, y, flags, param): 
    if event == cv2.EVENT_LBUTTONDOWN :
        print("mouse click, x:", x ," y:", y)
        global coord_x
        coord_x = x		
        global coord_y
        coord_y = y
	
def xy_uuid(x, y):
    return 'xy_%03d_%03d_%s' % (x, y, uuid1())

def imageCopy(src):
    return np.copy(src)

def videoProcess(openpath):

    DATASET_DIR = 'dataset_xy'
    
    try:
        os.makedirs(DATASET_DIR)
    except FileExistsError:
        print('Directories not created becasue they already exist')
    
    #cap = cv2.VideoCapture(openpath, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    cv2.setMouseCallback('Input', mouse_callback)
    linecolor = (255,0,0)

    print("click the road center and push 's' button to save labeled image.")

    try:
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                #frame = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                im = imageCopy(frame)
                im = cv2.line(im, (112, 0), (112, 224), linecolor, 3, cv2.LINE_AA)
                cv2.imshow("Input", im)
                #print(coord_x)

            else:
                break

            if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('s'):

                uuid = xy_uuid(coord_x, coord_y)
                filename = os.path.join(DATASET_DIR, uuid + '.jpg')

                print(filename)
                cv2.imwrite(filename, frame)

    except KeyboardInterrupt:  
        print("key int")
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)
        return	
	
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return
   
if __name__=="__main__":
    videoProcess(gst_str)
