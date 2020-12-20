# import the necessary packages
import imutils
import time
import argparse
import os
import cv2
import numpy as np
import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

from utils import loadModels, constructArgParser, extractBoxAndFaceROI


def detectAndPredictMask(frame, faceNet, faceMaskClassifier, threshold):
    # construct a blob from the image to pass to the network
    # using standard weights for the face detection model for image preprocessing
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # obtain the face detections by passing the blob through the network
    faceNet.setInput(blob)
    faceDetections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, faceDetections.shape[2]):
        # extract only confident detections using the confidence/probability
        # associated with the detection
        confidence = faceDetections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence 0.5 or input variable
        if confidence > threshold:
            # extract bounding box dimensions and face Region of intrest for classification
            # faceROI, startX, startY, endX, endY = extractBoxAndFaceROI(frame, faceDetections, itemNum=i,
            #                                                            height=h, width=w)

            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = faceDetections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all* faces at the same time
        # rather than one-by-one predictions in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = faceMaskClassifier.predict(faces, batch_size=32)

    # return face locations and their corresponding locations as a tuple
    return (locs, preds)


if __name__ == "__main__":
    # parse the arguments by constructing the argument parser
    args = constructArgParser()

    #loading the models form disk
    faceNet, faceMaskClassifier = loadModels(args["face"], args["face"], args["maskModel"])

    # initialize the video stream and allow the camera sensor to start
    print("starting the video stream windw to track faces and clasify them")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grabing a frame from the video stream and resizing the frame to 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a face mask or not
        (locs, preds) = detectAndPredictMask(frame, faceNet, faceMaskClassifier, threshold = args["confidence"])

        # loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # find the class and associated colour to use for the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability of prediction in the label of the bounding box
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame including the bounding box and label
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop to exit the program
        if key == ord("q"):
            break

