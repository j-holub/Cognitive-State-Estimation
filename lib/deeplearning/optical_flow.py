
from keras.models import Sequential
from keras.layers import Conv3D, ZeroPadding3D
from keras.layers import MaxPooling3D, GlobalMaxPool3D
from keras.layers import Dense

from keras import optimizers

def of_network():

    model = Sequential()

    model.add(ZeroPadding3D(padding=(2, 1, 1)))
    model.add(Conv3D(16, kernel_size=(8,3,3), activation='relu', strides=(4,2,2), input_shape=(64,128,128,3)))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1,2,2)))

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(32, kernel_size=(3,3,3), activation='relu', strides=(1,1,1)))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2)))

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(64, kernel_size=(3,3,3), activation='relu', strides=(1,1,1)))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2)))

    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(Conv3D(128, kernel_size=(4,3,3), activation='relu', strides=(1,1,1)))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1,2,2)))

    model.add(GlobalMaxPool3D())

    model.add(Dense(1024, activation='relu'))

    model.add(Dense(512, activation='relu'))

    model.add(Dense(5, activation='softmax'))


    return model
