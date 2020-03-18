import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D
from tensorflow.keras.activations import relu, softmax


class C3D:
    def __init__(self, frames, width, height, dimension, class_len):
        self.dimension = (frames, width, height, dimension)
        self.output = class_len

    def getModel(self, weights):
        model = Sequential()
        # 1st layer group and Input Layer
        model.add(Conv3D(64, (3, 3, 3), activation=relu, padding='same', strides=(1, 1, 1), name='conv1',
                         input_shape=self.dimension))
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
        if weights:
            model.load_weights('../src/C3D_weights.hdf5', by_name=True)

        return model


model = C3D(16, 227, 227, 3, 8).getModel(True)
model.summary()

# Deep Feature Extractor
extractor_model = Sequential()
for layer in model.layers[:-2]:
    extractor_model.add(layer)

# Freeze layers
for layer in extractor_model.layers:
    layer.trainable = False

print(extractor_model.summary())

path = "../data/ravdess_visual_frames"
videos = os.listdir(path)
data = []
sizer = 0

for video in videos:
    video_name = path + "/" + video
    feature_extract = []
    for segment in np.load(video_name):
        feature_extract.append(extractor_model.predict(np.array([segment])))
    extract = np.array(feature_extract).reshape(np.array(feature_extract).shape[0], 4096)
    avg = np.apply_over_axes(np.average, extract, [0])
    data.append([video[:-4], avg, int(video[7:8]) - 1])
    sizer += 1
    if sizer % 10 == 0:
        print(sizer)

data = np.array(data)
df = pd.DataFrame(data, columns=['Filename', 'Visual Feature', 'Emotion'])
df.to_hdf("visual.h5", key="visual")
print(df.head())

"""
# Read
r = pd.read_hdf("visual.h5")
r.head()
print(r.head())
print(r['Visual Feature'][0].shape)


"""