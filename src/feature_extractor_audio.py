import os
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Conv1D, Activation, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def get_model():
    model = Sequential()

    model.add(Conv1D(128, 5, padding='same', input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(8))
    model.add(Activation('softmax'))

    return model


def x_y():
    filename = []
    x = []
    y = []
    path = "../data/ravdess_audio_frames_mfcc/"
    for file in os.listdir(path):
        filename.append(file[:-4])
        audio_name = path + file
        label = int(file[7:8]) - 1
        x.append(np.load(audio_name))
        y.append(label)
    x = np.array(x)
    return np.array(filename), np.expand_dims(x.reshape(x.shape[0], 40), axis=2), to_categorical(y)


if __name__ == '__main__':
    model = get_model()
    model.load_weights("Conv1d_weights.hdf5")

    # Deep Feature Extractor
    extractor_model = Sequential()
    for layer in model.layers[:-2]:
        extractor_model.add(layer)

    # Freeze layers
    for layer in extractor_model.layers:
        layer.trainable = False

    model = extractor_model
    print(model.summary())

    filename, x, y = x_y()
    data = []
    for i in range(filename.shape[0]):
        if filename[i][1] != '2':
            data.append([filename[i], model.predict(np.array([x[i]])), y[i]])

    data = np.array(data)
    df = pd.DataFrame(data, columns=['Filename', 'Audio Feature', 'Emotion'])
    df.to_hdf("audio.h5", key="audio")
    print(df.head())
    print(df.shape)
