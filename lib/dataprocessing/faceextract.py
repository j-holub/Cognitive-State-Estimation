import os

import cv2
import numpy as np



class FaceExtractor:
    """Class representing a module to extract faces from images

    This class uses opencv's dnn module with a pretrained caffe model to find
    faces in the frames from the videos recorded in the experiment.

    It requires the model and config file for the caffe model

    Methods:
        extractFace(frame, size)
            extracts the face from a frame resize to (size, size, 3)
    """


    def __init__(self):
        # path of this file
        file_path  = os.path.dirname(os.path.realpath(__file__))
        # Caffe moodel files
        modelFile  = os.path.join(file_path, '..', '..', 'resources', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
        configFile = os.path.join(file_path, '..', '..', 'resources', 'deploy.prototxt')

        # load the deep neural network pretrained for face detection
        self.__cnn = cv2.dnn.readNetFromCaffe(configFile, modelFile)

        # confidence threshold for the face detection
        self.__confidence_threshold = 0.6



    def extractFace(self, frame: np, size: int):
        """Extracts the face in a frame

        Given a numpy array of a frame, it will use opencv's dnn module to extract
        the face from the image and resize it to a squared image of dimension
        (size, size, 3), if a face was found

        Parameters:
            frame (np): numpy array of the frame from the videos recorded in the
                experiments
            size (int): size (width & height) the face image should be cropped to

        Returns:
            bool, ndarray: the bool indicates whether a face was found or not
                and the ndarray will contain the face in dimension (size, size, 3)
                if found or will be None otherwise
        """

        # transform it to a 300x300 blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        # feed it to the network
        self.__cnn.setInput(blob)
        detections = self.__cnn.forward()

         # list with face detections with a high enough confidence
        detections = [detections[0, 0, i] for i in range(detections.shape[2]) if detections[0, 0, i, 2] > self.__confidence_threshold]
        print(len(detections))
        # only extract a face if there was any face found
        if(len(detections) > 0):
            # height and width of the frame
            h,w = frame.shape[:2]

            detect = detections[0]
             # get the corner coordinates for the rectangle
            x1 = int(detect[3] * w)
            y1 = int(detect[4] * h)
            x2 = int(detect[5] * w)
            y2 = int(detect[6] * h)

            # compute width and height of the face frame
            width  = np.abs(x2-x1)
            height = np.abs(y2-y1)

            # make the image squared
            diff = np.abs(height-width)
            if height > width:
                x1 -= int(np.floor(diff/2.0))
                x2 += int(np.ceil(diff/2.0))
            else:
                y1 -= int(np.floor(diff/2.0))
                y2 += int(np.ceil(diff/2.0))

            # crop the image
            cropped_image = frame[y1:y2, x1:x2]
            c_h, c_w = cropped_image.shape[:2]

            assert c_h == c_w

            # resize the image to 128 pixels
            cropped_image = cv2.resize(cropped_image, (128, 128))

            return (True, cropped_image)

        # no face was found
        else:
            return (False, None)
