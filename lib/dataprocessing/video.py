import datetime
import os

import cv2
import numpy as np

from .eyeextract import EyeExtractor
from .faceextract import FaceExtractor
from .opticalflow import OpticalFlow

class VideoHandler:
    """
    Class representing a video handler, that stores a video from the experiment
    and offers methods to extract information from it.

    It keeps a reference to the video which can be accessed when needed

    Methods:
        set_video(video_path)
            opens a new video
        get_frames(start, end, crop)
            retrieves the frames between two timestamps
        get_optical_flow_frames(start, end, crop)
            retrieves the frames between two timestamps, cropped to the face,
            and computes the optical flow between consecutive
            frames
        get_eye_frames(start, end, crop)
            retrieves the frames between two timestamps, cropped to the right
            eye, and computes the optical flow between consecutive
            frames
    """



    def __init__(self, video_path: str):
        """
        Parameters:
            video_path (str): path to the video file recorded in the experiment
        """
        # store the creating time which is the starting time of the video
        self.__video_start = self.__read_timestamp_from_filename(os.path.basename(video_path))

        # video capture stream
        self.__cap = cv2.VideoCapture(video_path)

        # eye extraction
        self.__ee = EyeExtractor()

        # face extraction
        self.__fe = FaceExtractor()

        # optical flow module
        self.__of = OpticalFlow()



    def set_video(self, video_path: str):
        """Opens a new video file

        Parameters:
            video_path (str): the path to the video file
        """

        # close the video stream if it's opened
        if(self.__cap.isOpened()):
            self.__cap.release()
        # open the new video
        self.__cap.open(video_path)



    def get_frames(self, start: datetime, end: datetime, crop: int):
        """Returns the frames cropped to the face between two timestamps

        Given the start timestamp and end timestamp as datetime objects, this
        method will return a numpy array with all the frames inbetween, cropped
        to the face

        Parameters:
            start (datetime):
                start timestamp
            end (datetime):
                end timestamp
            crop (int):
                size the face image should be cropped to (squared)

        Returns:
            ndarray: numpy array of the shape (frames, crop, crop)
        """

        # calculate the timestamps for the boundaries
        start_timestamp = ((start - self.__video_start).total_seconds() * 1000)
        end_timestamp   = ((end   - self.__video_start).total_seconds() * 1000)


        # get the framenumber for the last timestamp
        self.__cap.set(cv2.CAP_PROP_POS_MSEC, end_timestamp)
        end_frame = self.__cap.get(cv2.CAP_PROP_POS_FRAMES)


        # set the video to the beginning and get the frame number
        self.__cap.set(cv2.CAP_PROP_POS_MSEC, start_timestamp)
        frame_pos = self.__cap.get(cv2.CAP_PROP_POS_FRAMES)

        # numpy array of frames
        frames = np.zeros([1,crop,crop], dtype=np.uint8)

        # iterate over all the frames
        while(frame_pos <= end_frame):
            success, frame = self.__cap.read()
            # if the frame could be read
            if(success):
                found, face = self.__fe.extractFace(frame, crop)
                # if a face was found
                if(found):
                    # convert the frame to grayscale
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                    # add it to all the frames
                    frames = np.concatenate((frames, np.expand_dims(face, axis=0)), axis=0)
                    # increment the frame number

            frame_pos = frame_pos+1

        # return all the frames minus the first 0 frame
        return frames[1:,...]



    def get_optical_flow_frames(self, start: datetime, end: datetime, crop: int):
        """Returns the frames cropped to the face but as optical flow images

        Optical flow images encode the information about what has changed between
        two frames with color and intensity using the HSV color space.

        Parameters:
            start (datetime):
                start timestamp
            end (datetime):
                end timestamp
            crop (int):
                size the face image should be cropped to (squared)

        Returns:
            ndarray: numpy array of the shape (frames, crop, crop, 3)
        """

        # compute the grayscale frames for the given timespan
        facial_frames = self.get_frames(start, end, crop)

        # optical flow frames
        of_frames = np.zeros([1, crop, crop, 3], dtype=np.uint8)

        for i in range(1, len(facial_frames)):
            # get the two frames to compute the optical flow for
            prev = facial_frames[i-1]
            curr = facial_frames[i]

            # compute the optical flow image between the frames
            of_image = self.__of.optical_flow(prev, curr)

            # add the frames to the matrix
            of_frames = np.concatenate((of_frames, np.expand_dims(of_image, axis=0)), axis=0)


        return of_frames[1:,...]


    def get_eye_frames(self, start: datetime, end: datetime, crop: int):
        """Returns the frames cropped to the eye between two timestamps

        Given the start timestamp and end timestamp as datetime objects, this
        method will return a numpy array with all the frames inbetween, cropped
        to the eye

        Parameters:
            start (datetime):
                start timestamp
            end (datetime):
                end timestamp
            crop (int):
                size the face image should be cropped to (squared)

        Returns:
            ndarray: numpy array of the shape (frames, crop, crop)
        """

        # calculate the timestamps for the boundaries
        start_timestamp = ((start - self.__video_start).total_seconds() * 1000)
        end_timestamp   = ((end   - self.__video_start).total_seconds() * 1000)


        # get the framenumber for the last timestamp
        self.__cap.set(cv2.CAP_PROP_POS_MSEC, end_timestamp)
        end_frame = self.__cap.get(cv2.CAP_PROP_POS_FRAMES)


        # set the video to the beginning and get the frame number
        self.__cap.set(cv2.CAP_PROP_POS_MSEC, start_timestamp)
        frame_pos = self.__cap.get(cv2.CAP_PROP_POS_FRAMES)

        # numpy array of frames
        frames = np.zeros([1,crop,crop], dtype=np.uint8)

        # iterate over all the frames
        while(frame_pos <= end_frame):
            success, frame = self.__cap.read()
            # if the frame could be read
            if(success):
                # found, eye = self.__ee.get_eye(frame, 64)
                found, eye = self.__ee.extractEye(frame, 64)
                if(found):
                    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                    frames = np.concatenate((frames, np.expand_dims(eye, axis=0)), axis=0)

            frame_pos += 1

        return frames[1:,...]



    def __read_timestamp_from_filename(self, filename: str):
        """Computes the timestamp from the video file name

        Parameters:
            filename (str):
                the name of the video file that contains the timestamp.
                It must be in the format YY-MM-DD_-hh-mm-ss

        Returns:
            datetime: datetime object of the timestamp
        """

        timestamp = [int(date) for date in
            os.path.splitext(filename.replace('_', '-'))[0].split('-')
        ]
        return datetime.datetime(*timestamp)
