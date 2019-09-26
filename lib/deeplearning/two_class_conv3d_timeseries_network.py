from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers import MaxPooling3D, GlobalMaxPool3D
from keras.layers import Dense

def twoclass_CLitW_network(windowsize: int = 60):

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

    model.add(Dense(2, activation='softmax'))

    return model
