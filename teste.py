from os.path import expanduser
import os.path as osp
import sys,os
import numpy as np

homedir = "~/laeonetplus-main/"
 
sys.path.insert(0, os.path.join(homedir,"utils")) # CHANGE ME
# sys.path.insert(0, os.path.join(homedir,"datasets")) # CHANGE ME
sys.path.insert(0, os.path.join(homedir,"tracking")) # CHANGE ME
 
import socket

#os.environ["CUDA_VISIBLE_DEVICES"]="0" # "-1"
#gpu_rate = 0.99 # CHANGE ME!!!

#theSEED = 1330
  
# for reproducibility
#np.random.seed(theSEED)
 
from utils.mj_tracksManager import TracksManager
#from ln_avagoogleImages import mj_getImagePairSeqFromTracks, mj_getFrameBBsPairFromTracks
#from ln_laeoImage import mj_padImageTrack
from tracking.ln_tracking_heads import process_video
 
from tensorflow.keras.models import load_model


import cv2

from utilstr.utils import DetectionParameters

det_params = DetectionParameters()
det_params.MinScore = 0.25  # Change me 

def detect_on_image(frame, model, verbose=False):

    # Begin detection
    dets_dict = {}    
    print('inicio detect#########')

    # Adapt image to networks format
    org_resolution = np.flip(frame.shape[:2]) # reverse to width, height format
    frame = cv2.resize(frame, (det_params.InputWidth, det_params.InputHeight))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform prediction
    y_pred = model.predict(np.expand_dims(frame, axis=0))
    y_pred = y_pred[0]
    # Filter out detections below threshold
    y_pred = y_pred[y_pred[:, 1] > det_params.MinScore]
    # Rearrange to format [xmin, ymin, xmax, ymax, score, class]
    y_pred = y_pred[:, np.array([2, 3, 4, 5, 1, 0])]
    # Scale back detections to original frame size
    y_pred[:, :4] = y_pred[:, :4] * np.tile(org_resolution, 2) / det_params.InputWidth
    if verbose > 1:
        print(y_pred)
    dets_dict = y_pred

    return dets_dict
    
 
import requests
# import os
from ln_tracking_heads import load_detection_model

model = load_detection_model()

#image = cv2.imread('image.jpg')

	
#cv2.imshow('fim#########',image)

#cv2.waitKey()


#import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow('teest')

img_counter = 0

while True:
	ret, frame = cam.read()
	
	frame = cv2.resize(frame, (int(frame.shape[1]*0.2), int(frame.shape[0]*0.2)), interpolation = cv2.INTER_AREA)
	if not ret:
		break
		
	dets = detect_on_image(frame, model, verbose=True)

	for bb in dets:
		xmin = int(bb[0])
		ymin = int(bb[1])
		xma = int(bb[2])
		yma = int(bb[3])
		cv2.rectangle(frame,(xmin, ymin), (xma, yma),(0,0,255))
		
	cv2.imshow("teest", frame)
	
	k = cv2.waitKey(1)
	
	if k%256 == 27:
		print("exit")
		break

cam.release()

cv2.destroyAllWindows()


print('fim#########')
    
   

