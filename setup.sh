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
wget \
	 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
	 -O resources/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 resources/shape_predictor_68_face_landmarks.dat.bz2

# remove the unneeded data
rm -r openface-master master.zip

cd ${ORIG_DIR}
