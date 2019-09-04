import os

import cv2
import numpy as np
import openface

from .faceextract import FaceExtractor


class EyeExtractor:
    """Class representing a module to extract the right eye from images

    This classes uses OpenFace to find facial landmarks and then extracts the
    right eye from the image

    It requires the 'shape_predictor_68_face_landmarks.dat' file from dlib

    Methods:
        extractEye(frame, size)
            extracts the face from a frame resize to (size, size, 3)
    """

    def __init__(self):

        # path of this file
        file_path  = os.path.dirname(os.path.realpath(__file__))
        # load the dlib model for facial landmarks
        self.__align = openface.AlignDlib(os.path.join(file_path, '..', '..', 'resources', 'shape_predictor_68_face_landmarks.dat'))


    def extractEye(self, frame: np.ndarray, crop: int):
        """Extracts the right eye from an image

        Using OpenFace this method finds facial landmarks and extracts the right
        eye cropped to cropxcrop pixels

        Parameters:
            frame (np.ndarray):
                the video frame that should be processed
            crop (int):
                final crop size for the eye image

        Returns:
            bool, np.ndarray: the bool indicates wether the eye was found and if
                the np.ndarray array of size (crop, crop, 3) contains the cropped
                eye image and is None otherwise
        """

        # find the face and then get the landmarks
        bb = self.__align.getLargestFaceBoundingBox(frame)

        # if a face bounding box was found extract the eye
        if(bb):
            # get the facial landmarks
            landmarks = self.__align.findLandmarks(frame,bb)

            # relevant landmarks for the right eye
            right_eye_outer = landmarks[36]
            right_eye_inner = landmarks[39]

            # compute the x center of the eye
            middle_x = int(np.abs(right_eye_inner[0]-right_eye_outer[0])/2)
            center_x = int(right_eye_outer[0]+ middle_x)
            # compute the y center of the eye
            middle_y = int(np.abs(right_eye_inner[1]-right_eye_outer[1])/2)
            center_y = int(right_eye_inner[1]+middle_y) if right_eye_inner[1] < right_eye_outer[1] else int(right_eye_outer[1]+middle_y)

            # extract the eye from the frame
            eye_img = frame[center_y-middle_x*2:center_y+middle_x*2,center_x-middle_x*2:center_x+middle_x*2,...]

            return True, cv2.resize(eye_img, (crop, crop))

        # return None if no face was found in the frame
        else:
            return False, None
