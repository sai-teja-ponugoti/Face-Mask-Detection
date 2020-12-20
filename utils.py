import os
import cv2
from tensorflow.keras.models import load_model

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