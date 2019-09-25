from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers import MaxPooling3D, GlobalMaxPool3D
from keras.layers import Dense, Flatten

def CLitW_network(windowsize: int = 60):
    """The neural network as proposed in 'Cognitive Load in the Wild'

    This CNN was proposed for 0-back / 1-back / 2-back classification using
    sequences of eye images

    This network has 5 neurons on the output layer for 5 class classification

    Parameters:
        windowsize (int):
            width of one chunk that is input to the network
            default: 60

    Returns:
        keras model: the model object storing the information about the network
    """

    model = Sequential()

    model.add(Conv3D(128, (3,3,3), activation='relu', input_shape=(windowsize,64,64,1)))

    model.add(Conv3D(128, (3,3,3), activation='relu'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2)))

    model.add(Conv3D(128, (3,3,3), activation='relu'))

    model.add(Conv3D(128, (3,3,3), activation='relu'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2)))

    model.add(Conv3D(128, (3,3,3), activation='relu'))

    model.add(Conv3D(128, (3,3,3), activation='relu'))

    model.add(Conv3D(128, (3,3,3), activation='relu'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2)))

    model.add(GlobalMaxPool3D())

    model.add(Dense(1024, activation='relu'))

    model.add(Dense(512, activation='relu'))

    model.add(Dense(5, activation='softmax'))

    return model
