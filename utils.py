# import the necessary packages
import os
import cv2
import numpy as np
import argparse

import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def constructArgParser():
    # parse the arguments by constructing the argument parser
    argP = argparse.ArgumentParser()
    argP.add_argument("-i", "--testImagePath", type=str, default="test_image.jpg",
                    help="path to input image")
    argP.add_argument("-f", "--face", type=str,
                    default="models",
                    help="path to face detector model directory")
    argP.add_argument("-m", "--maskModel", type=str,
                    default="models/face_mask_classifier.model",
                    help="path to trained face mask detector model")
    argP.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(argP.parse_args())

    return args

def loadModels(arch_path, weights_path, face_mask_path):
    """
    Utility function to load pre-trained face detection and trained face ma classifier
    :param arch_path: path of pre-trained face detection model architecture
    :param weights_path: path of pre-trained face detection model weights
    :param face_mask_path: path of face mask classification model
    :return: return faceNet and faceMaskClassifier models
    """
    # loading the pretrained face detection OpenCV2 caffe model
    print("loading face detector model...")
    # model architecture (i.e., the layers) file path
    prototxtPath = os.path.sep.join([arch_path, "deploy.prototxt"])
    # model weights file path
    weightsPath = os.path.sep.join([weights_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    # loading the pretrained face detection model using above architecture ad weights
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask classifier built using dev_train_mask_detector file
    print("loading face mask detector model...")
    faceMaskClassifier = load_model(face_mask_path)

    return faceNet, faceMaskClassifier


def extractBoxAndFaceROI(image, faceDetections, itemNum, height, width):
    # compute the (x, y)-coordinates of the bounding box for the face
    boundingBox = faceDetections[0, 0, itemNum, 3:7] * np.array([width, height, width, height])
    (startX, startY, endX, endY) = boundingBox.astype("int")

    # making sure bounding box is inside the dimensions of the image
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

    # extract the face region of interest(ROI), convert it from BGR to RGB channel
    # ordering, resize it to 224x224, and preprocessing it to be compatible with
    # face mask classification model
    faceROI = image[startY:endY, startX:endX]
    faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2RGB)
    faceROI = cv2.resize(faceROI, (224, 224))
    faceROI = img_to_array(faceROI)
    faceROI = preprocess_input(faceROI)

    return faceROI, startX, startY, endX, endY


