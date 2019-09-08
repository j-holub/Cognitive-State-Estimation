# get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# save the original working dir
ORIG_DIR=$(pwd)
# switch to the script dir
cd ${SCRIPT_DIR}

# setup virtual environment
python3 -m venv env
# activate it
source env/bin/activate

# install the basic dependencies
pip3 install -r requirements.txt

# get the OpenFace git repository
wget https://github.com/cmusatyalab/openface/archive/master.zip
unzip master.zip
# install openface using pip
pip3 install ./openface-master

# get the dlib landmarks file for the facial landmark detection
cd ${SCRIPT_DIR}
mkdir -p resources

# open face landmark detector model
wget \
	 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
	 -O resources/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 resources/shape_predictor_68_face_landmarks.dat.bz2

# caffe model for opencv face detection
wget \
	https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/res10_300x300_ssd_iter_140000_fp16.caffemodel \
	-O resources/res10_300x300_ssd_iter_140000_fp16.caffemodel

wget \
	https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt \
	-O resources/deploy.prototxt


# remove the unneeded data
rm -r openface-master master.zip

cd ${ORIG_DIR}
