import cv2 as cv
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D
from tensorflow.keras.activations import relu, softmax

emotions = {
    0: "angry",
    1: "calm",
    2: "disgust",
    3: "fearful",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprised"
}


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
        if weights == True:
            model.load_weights('/content/drive/My Drive/sports1M_weights.h5', by_name=True)

        return model


class VideoParser:
    def __init__(self, width, height):
        self.dimension = (width, height)
        self.face_cascade = cv.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')

    def readVideo(self, path):
        frame_list = []
        capture = cv.VideoCapture(path)
        while (capture.isOpened()):
            ret, frame = capture.read()
            if (ret):
                faces = self.face_cascade.detectMultiScale(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), 1.3, 5)
                for (x, y, w, h) in faces:
                    frame = frame[y: y + h, x: x + w]
                    if (frame.size != 0):
                        frame = cv.resize(frame, self.dimension, interpolation=cv.INTER_AREA)
                        frame_list.append(frame)
            else:
                capture.release()
                break
        return np.array(frame_list)


class Frame:
    def __init__(self, frame_size, width, height, dimension):
        self.frame_size = frame_size
        self.width = width
        self.height = height
        self.dimension = dimension

    def getFrame(self, data):
        data = np.array(data)
        if data.shape[0] % self.frame_size == 0:
            final = data.reshape(int(data.shape[0] / self.frame_size), self.frame_size, self.width, self.height,
                                 self.dimension)
        else:
            overlap = (data.shape[0] - self.frame_size * int(data.shape[0] / self.frame_size))  # remain frame size / 16
            beginning = int((self.frame_size - overlap) / 2)  # (16 - L) / 2
            ending = self.frame_size - overlap - beginning
            final = np.concatenate((data[:-overlap], np.concatenate(
                (data[:beginning], np.concatenate((data[-overlap:], data[-ending:]))))))
            final = final.reshape(int(final.shape[0] / self.frame_size), self.frame_size, self.width, self.height,
                                  self.dimension)

        return final


def preprocessing(video_path):
    return Frame(16, 227, 227, 3).getFrame(VideoParser(227, 227).readVideo(video_path))


def get_model():
    model = C3D(16, 227, 227, 3, 8).getModel(False)
    model.load_weights("./weights/C3D_weights.hdf5")
    return model


"""
def display(predict_video):
    result = ""
    predict_list = np.argmax(predict_video, axis=1)
    for i in range(len(predict_list)):
        result += "Segment {:2}: {} ".format(i + 1, emotions.get(predict_list[i]))
    return result
"""


def display(predict_video):
    result = []
    predict_list = np.argmax(predict_video, axis=1)
    for i in range(len(predict_list)):
        result.append("Segment {:2}: {} ".format(i + 1, emotions.get(predict_list[i])))
    return result


"""
model = get_model()
video = preprocessing("/content/drive/My Drive/merve-angry.mp4")
predictVideo = model.predict(video, batch_size=2)
display(predictVideo, "angry")

video = preprocessing("/content/drive/My Drive/furkan-happy.mp4")
predictVideo = model.predict(video, batch_size=2)
display(predictVideo, "happy")
"""
