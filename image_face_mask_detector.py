# import the necessary packages
import os
import cv2
import numpy as np
import argparse

import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from utils import loadModels, constructArgParser, extractBoxAndFaceROI

def detectFaceAndClassify(faceNet, faceMaskClassifier, testImagePath, threshold):
    # load the input test image from disk
    image = cv2.imread(testImagePath)
    # making a copy of image and finding the image spatial dimensions
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image to pass to the network
    # using standard weights for the face detection model for image preprocessing
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # obtain the face detections by passing the blob through the network
    print("computing face detections...")
    faceNet.setInput(blob)
    faceDetections = faceNet.forward()

    # loop over the detections to classify them and form bounding boxes and labels
    for i in range(0, faceDetections.shape[2]):
        # extract only confident detections using the confidence/probability
        # associated with the detection
        confidence = faceDetections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence 0.5 or input variable
        if confidence > threshold:
            # extract bounding box dimensions and face Region of intrest for classification
            faceROI, startX, startY, endX, endY = extractBoxAndFaceROI(image, faceDetections, itemNum=i,
                                                                       height=h, width=w)

            faceROI = np.expand_dims(faceROI, axis=0)

            # Passing the pre-processed image with classification model to check if there is a mask or not
            (mask, withoutMask) = faceMaskClassifier.predict(faceROI)[0]
            # (mask, withoutMask) = faceMaskClassifier.predict(faceROI)

            # find the class and associated colour to use for the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability of prediction in the label of the bounding box
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # forming bounding box rectangle and display the label the output image frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # show the output image
    cv2.imshow("Output", image)
    # display the image still a key is pressed, when key is pressed program is terminated
    cv2.waitKey(0)


if __name__ == "__main__":
    # parse the arguments by constructing the argument parser
    args = constructArgParser()

    #loading the models form disk
    faceNet, faceMaskClassifier = loadModels(args["face"], args["face"], args["maskModel"])

    # call to function detectFaceAndClassify to detect and classify faces with and with out mask
    detectFaceAndClassify(faceNet, faceMaskClassifier, testImagePath=args["testImagePath"], threshold = args["confidence"])
