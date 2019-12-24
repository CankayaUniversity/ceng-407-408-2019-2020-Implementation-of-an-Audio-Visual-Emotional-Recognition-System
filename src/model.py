from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D
from tensorflow.keras.activations import relu, softmax

class C3D:
    def __init__(self, frames, width, height, dimension, class_len):
        self.dimension = (frames, width, height, dimension)
        self.output = class_len
    
    def getModel(self):
        model = Sequential()
        # 1st layer group and Input Layer
        model.add(Conv3D(64, (3,3,3), activation=relu, padding='same', strides=(1, 1, 1), name='conv1', input_shape=self.dimension))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', strides=(1, 2, 2), name='pool1'))

        # 2nd layer group
        model.add(Conv3D(128, (3, 3, 3), activation=relu, padding='same', strides=(1, 1, 1), name='conv2'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='valid', strides=(2, 2, 2), name='pool2'))

        # 3rd layer group
        model.add(Conv3D(256, (3, 3, 3), activation=relu, padding='same', strides=(1, 1, 1), name='conv3a'))
        model.add(Conv3D(256, (3, 3, 3), activation=relu, padding='same', strides=(1, 1, 1), name='conv3b'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='valid', strides=(2, 2, 2), name='pool3'))

        # 4th layer group
        model.add(Conv3D(512, (3, 3, 3), activation=relu, padding='same', strides=(1, 1, 1), name='conv4a'))
        model.add(Conv3D(512, (3, 3, 3), activation=relu, padding='same', strides=(1, 1, 1), name='conv4b'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='valid', strides=(2, 2, 2), name='pool4'))

        # 5th layer group
        model.add(Conv3D(512, (3, 3, 3), activation=relu, padding='same', strides=(1, 1, 1), name='conv5a'))
        model.add(Conv3D(512, (3, 3, 3), activation=relu, padding='same', strides=(1, 1, 1), name='conv5b'))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='valid', strides=(2, 2, 2), name='pool5'))
        model.add(Flatten())

        # Fully Connected Layer
        model.add(Dense(4096, activation=relu, name='fc6'))
        model.add(Dense(4096, activation=relu, name='fc7'))

        # Output Layer
        model.add(Dense(self.output, activation=softmax, name='fc8'))

        model.load_weights('../data/model/sports1M_weights.h5', by_name=True)

        return model
