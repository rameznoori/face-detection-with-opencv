import numpy as np
import argparse
import cv2

arg_p = argparse.ArgumentParser()
arg_p.add_argument("-i", "--image", required=True, help="Path to input image")
arg_p.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'deploy' prototxt file")
arg_p.add_argument("-m", "--model", required=True, help="Path to Caffe pre-trained model")
arg_p.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(arg_p.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (400, 400)), 1.0, (400, 400), (104.0, 177.0, 123.0))
