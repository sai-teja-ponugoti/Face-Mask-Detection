# Face-Mask-Detection
Face Mask Detection system both on image and video based on computer vision and deep learning using OpenCV and Tensorflow/Keras.

## Problem Statement:
Due to COVID-19, the usage of masks has become an atmost important precaution to stop spreading the virus and even protect us from getting infected. But lot of people tend to not use masks in public places which increases the chance of spreading the virus and make its containment more difficult. Restricting people without masks in public places and imposing strict rules in public transportation, gatherings, shopping malls, industries is an imprtant task to be acarried out. Inorder to do this we need to find people without masks in public. This project helps in finding the people without masks in static images as well as in real-time video streams, which can help in restricting that people to create hotspots in public.

## Dataset:
The dataset used consists of images of people with a mask and without a mask.

This dataset consists of __3835 images__ belonging to two classes:
*	__without_mask: 1919 images__
*	__with_mask: 1916 images__

The dataset used can be downloaded here - [Click to Download](https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG?usp=sharing)



## Frameworks used:
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

All the required frameworks can be installed using requirements.txt

## Development:

Refer to Jupyter notebooks in "dev jupyter notebooks" folders to understand step-by-step procedure for developing this project.

### Training:
Refer to dev_train_mask_detctor.ipynb ([See Python script](https://github.com/sai-teja-ponugoti/Face-Mask-Detection/blob/main/dev%20jupyter%20notebooks/dev_train_mask_detctor.ipynb)) to train the face mask classification model using the above mentioned dataset.

### Static Image face mask detection:
A pre-trained openCV face detection model has been used to detect the faces in the input image and extract face ROI(region of intrest) and pass it to face mask classifier to check whether a mask is worn or not. Refer to dev_train_mask_detctor.ipynb ([See Python script](https://github.com/sai-teja-ponugoti/Face-Mask-Detection/blob/main/dev%20jupyter%20notebooks/dev_train_mask_detctor.ipynb)) to follow the development process.

### Face mask detection in video stream:
Frames are continiously extracted from the video stream. The same pre-trained openCV face detetcion model is used for detecting faces in each frame and the detected face ROI are passed to face mask classifier to check whether a mask is worn or not. Refer to dev_Video_face_mask_detecion.ipynb ([See Python script](https://github.com/sai-teja-ponugoti/Face-Mask-Detection/blob/main/dev%20jupyter%20notebooks/dev_Video_face_mask_detecion.ipynb)) to follow the development process.

## Running files:
All ipynb files are converted to properly documented .py files can be run using the folling commands. Adjust the paths in the program as required.

1. Image face mask detection: Open terminal and execute the following command:
```
$ python3 image_face_mask_detector.py  --testImagePath "test image path" --face "folder which contains pre-trined models" --maskModel "mask classifier model path" --confidence "confidence of detected face"
```

2. Video face mask detection: To detect face masks in the video stream type the following command: 
```
$ python3 video_face_mask_detector.py --face "folder which contains pre-trined models" --maskModel "mask classifier model path" --confidence "confidence of detected face"
```

## Results:
### Face mask detection in image:

##### Input image: 
![](https://github.com/sai-teja-ponugoti/Face-Mask-Detection/blob/main/test_image.png)

##### Output image: 
![](https://github.com/sai-teja-ponugoti/Face-Mask-Detection/blob/main/output.png)

### Face mask detection in video stream:
![Live Demo](https://github.com/sai-teja-ponugoti/Face-Mask-Detection/blob/main/video_detection_demo.gif)

#### Check out the demo [here.](https://youtu.be/dN7FhmPzuxA)
