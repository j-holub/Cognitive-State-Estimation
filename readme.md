# Cognitive Load Estimation

This repository holds the code I used for my master thesis **Estimating Cognitive Load Using Deep Learning Methods**, supervised by Prof. Dr. Andreas Dengel and Dr. Shoya Ishimaru at the DFKI Kaiserslautern.

In the thesis we proposed different approaches based on deep learning approaches, that work with visual data, to estimate the cognitive load of a human. To evaluate these approaches we conducted a psychological experiment, namely the n-back experiment to record data. In this experiment the participants have to solve the same task in 5 different difficulty levels 1-5 (n). We trained deep learning models to estimate the difficulty level n from the facial and eye movements because past research has linked these movements to the cognitive state of a human.


## Setup

The `setup.sh` script performs all the neccessary setup steps for you. It does the following steps

* create a virtual environment in `env/` and activate it
* install the neccessary python dependencies stated in the `requirements.txt` file
* downloads and installs [Openface](https://github.com/cmusatyalab/openface)
* downloads the [Openface](https://github.com/cmusatyalab/openface) facial landmarks detector model
* downloads the face detection caffee model for [OpenCV](https://opencv.org)

Of course you can do all of this by hand if you prefer, just have a look into the `setup.sh` script

## Experiment

The code for the N-Back Experiment can be found [here](https://github.com/00SteinsGate00/N-Back-Experiment).


## Scripts

This repository contains various python scripts. Their usage is well documented within the script file itself. To see what arguments they take just open a script file and look at the docstring on the top.

The following scripts are available

* **process_data_raw**

  This script takes the recorded video and the experiment results (with timestamps) as an input. It extracts the relevant chunks of video, where the participant was performing the n-back trial. Depending on the method chose, frame by frame, either the face, the eye or an optical flow image is extraced. The script outputs the frames as numpy .npy files, one file for each trial. Alongside that file it outputs a .json file with the ground truth data

* **raw_to_training_data**

  This script transforms the output generetad by the **process_data_raw** script into training and validation sets, such that the models can be trained with the data. It couples the frames into chunks of a certain windowsize and can subsample the framerate if needed. The reason for the seperation of the two scripts was, that this makes it easier to test different subsampling rates and windowsizes, without having to expensively recomputed everything from the orginal video data
  It can output the data as a balanced data set for a single person, such that 4 trials of each difficulty level are taken as the training set and 1 is taken for the validation set

* **network**

  This script is the interface to all the networks of our proposed approaches. It is used to start training sessions and with the first argument you specify the neural network that should be used. It saves the models from the epoch with the best performance on the training and on the validation set and outputs a training history object

* **statistic**

  This script provides various statistics about the experiment data, such as the average score, the standard deviation and more

* **metric**

  This script is used to create all kinds of plots, that show how the approaches performed, how the accuracy and loss change over the course of the training and more


* **confusionmatrix**

  Computes the confusion matrix for a given model and the validation data plus ground truth labels. The matrix is output as a .npy file

* **lecture_video_predictions**

  Computes the predictions for the lecture video segment of the experiment, where the participants were just watching a lecture video to see, how their cognitive load changes
