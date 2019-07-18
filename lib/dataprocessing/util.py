import os

"""Package with utility functions for the processing of the experiment data

Offers functions to retrieve the files from an experiment directory

Functions:
    getExperimentDataFile(exp_data_path):
        retrieves the JSON file from the experiment directory
    getVideoFile(exp_data_path):
        retrieves the video file from the experiment directory
    getParticipantID(exp_data_path):
        reads the directory name as the participant ID
    getExperimentInfo(exp_data_path):
        retrieves the JSON file, the video file and the participant ID

"""

def getExperimentDataFile(exp_data_path: str):
    """Finds the experiment data json file in an experiment data folder

    Parameters:
        exp_data_path (str):
            path to the experiment data containing the video file

    Returns:
        str: absolute path to the json data file

    Raises:
        AssertionError
            if the number of json files is not exactly one
    """
    exp_data = [file for file in os.listdir(exp_data_path) if os.path.splitext(file)[1] == '.json']
    assert len(exp_data) == 1
    return os.path.join(exp_data_path, exp_data[0])


def getVideoFile(exp_data_path: str):
    """Finds the video file in an experiment data folder

    Parameters:
        exp_data_path (str):
            path to the experiment data containing the video file

    Returns:
        str: absolute path to the video

    Raises:
        AssertionError
            if the number of video files (.mp4) is not exactly one
    """

    videos = [file for file in os.listdir(exp_data_path) if os.path.splitext(file)[1] == '.mp4']
    assert len(videos) == 1
    return os.path.join(exp_data_path, videos[0])



def getParticipantID(exp_data_path: str):
    """Retrieves the participant id for the experiment data directory name

    Assumes the directory name to be the ID of the participant

    Parameters:
        exp_data_path (str):
            path to the experiment data containing the video file

    Returns:
        str: participant ID

    """

    return os.path.basename(exp_data_path)



def getExperimentInfo(exp_data_path: str):
    """Retrieves all the information for a participant data directory

    Gets the participant ID, video file path and json file path

    Parameters:
        exp_data_path (str):
            path to the experiment data. Must contain the JSON file from the
            JsPsych N-Back experiment and the video file recorded during the
            experiment

    Returns:
        str, str, str:
            tuple with the experiment data file path, the video file path and the
            ID of the participant
    """

    return \
        getExperimentDataFile(exp_data_path),\
        getVideoFile(exp_data_path),\
        getParticipantID(exp_data_path)
