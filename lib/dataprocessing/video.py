import datetime
import os

import cv2
import numpy as np

class VideoHandler:
    """
    Class representing a video handler, that stores a video from the experiment
    and offers methods to extract information from it.

    It keeps a reference to the video which can be accessed when needed

    Methods:
        get_frames(start, end)
            retrieves the frames between two timestamps
    """


    def __init__(self, video_path):
        """
        Parameters:
            video_path (str): path to the video file recorded in the experiment
        """
        # store the creating time which is the starting time of the video
        self.__video_start = datetime.datetime.fromtimestamp(os.path.getctime(video_path))

        # store the path the video
        self.__video_path = video_path

        # video capture stream
        self.__cap = cv2.VideoCapture(video_path)



    def get_frames(self, start: datetime, end: datetime):
        """Returns all frames between two timestamps

        Given the start timestamp and end timestamp as datetime objects, this
        method will return a numpy array with all the frames inbetween

        Parameters:
            start (datetime):
                start timestamp
            end (datetime):
                end timestamp

        Returns:
            ndarray: numpy array of the shape (frames, height, width)
        """

        # calculate the timestamps for the boundaries
        start_timestamp = ((start- self.__video_start).microseconds/1000)
        end_timestamp   = ((end  - self.__video_start).microseconds/1000)

        # get the framenumber for the last timestamp
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, end_timestamp)
        end_frame = self.__cap.get(cv2.CAP_PROP_POS_FRAMES)


        # set the video to the beginning and get the frame number
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, start_timestamp)
        frame_pos = self.__cap.get(cv2.CAP_PROP_POS_FRAMES)

        # numpy array of frames
        frames = np.zeros([1,1080,1920])

        while(frame_pos <= end_frame):
            check, frame = self.__cap.read()
            if(check):
                print(frame_pos)
                # convert the frame to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # add it to all the frames
                frames = np.concatenate((frames, np.expand_dims(frame, axis=0)), axis=0)
                # increment the frame number
                frame_pos = frame_pos+1

        # return all the frames minus the first 0 frame
        return frames[1:,...]
