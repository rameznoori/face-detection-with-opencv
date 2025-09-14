import numpy as np
import argparse
import cv2
#import packages for video/webcam detection
from imutils.video import VideoStream
import imutils
import time

arg_p = argparse.ArgumentParser()
arg_p.add_argument("-i", "--image", required=True, help="Path to input image")
arg_p.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'deploy' prototxt file")
arg_p.add_argument("-m", "--model", required=True, help="Path to Caffe pre-trained model")
arg_p.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(arg_p.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)), 1.0, (400,400), (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
#loop over detections
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence < args["confidence"]:
        continue
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (start_X,start_Y,end_X,end_Y) = box.astype("int")
    text = "{:.2f}%".format(confidence * 100)
    y = start_Y - 10 if start_Y - 10 > 10 else start_Y + 10
    cv2.rectangle(image, (start_X, start_Y), (end_X, end_Y), (0, 0, 255), 2)
    cv2.putText(image, text, (start_X, start_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (400, 400)), 1.0, (400, 400), (104.0, 177.0, 123.0))

print("[INFO] computing object detection...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_X, start_Y, end_X, end_Y) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)
        y = start_Y - 10 if start_Y - 10 > 10 else start_Y + 10
        cv2.rectangle(image, (start_X, start_Y), (end_X, end_Y), (0, 0, 255), 2)
        cv2.putText(image, text, (start_X, start_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)