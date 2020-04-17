import librosa
import numpy as np
import subprocess

from tensorflow.keras.models import Sequential
from .visual_model import preprocessing, get_model
from tensorflow.keras.layers import Conv1D, Activation, MaxPooling1D, Dropout, Flatten, Dense

emotions = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}


def audio_model():
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

    model.load_weights("./weights/audio_weights.hdf5")
    return model


def fusion_model():
    model = Sequential([
        Dense(4096, input_shape=(8192,)),
        Activation('relu'),
        Dropout(0.2),

        Dense(2048),
        Activation('relu'),
        Dropout(0.2),

        Dense(1096),
        Activation('relu'),
        Dropout(0.2),

        Dense(8),
        Activation('softmax')
    ])
    model.load_weights("./weights/fusion_weights.hdf5")
    return model


def feature_visual_model():
    model = get_model()

    # Deep Feature Extractor
    extractor_model = Sequential()
    for layer in model.layers[:-2]:
        extractor_model.add(layer)

    # Freeze layers
    for layer in extractor_model.layers:
        layer.trainable = False
    return extractor_model


def feature_audio_model():
    model = audio_model()
    # Deep Feature Extractor
    extractor_model = Sequential()
    for layer in model.layers[:-2]:
        extractor_model.add(layer)

    # Freeze layers
    for layer in extractor_model.layers:
        layer.trainable = False
    return extractor_model


def extract_audio(path):
    command = "ffmpeg -y -i {} -ab 160k -ac 2 -ar 44100 -vn ./media/contents/audio_extract.wav".format("." + path)
    subprocess.call(command, shell=True)


def preprocess_audio(path):
    extract_audio(path)
    X, sample_rate = librosa.load("./media/contents/audio_extract.wav", res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    return np.expand_dims(mfcc.reshape(1, 40), axis=2)


def feature_extract_visual(model, video):
    feature = []
    for segment in range(video.shape[0]):
        feature.append(model.predict(np.array([video[segment]])))
    extract_feature = np.array(feature).reshape(np.array(feature).shape[0], 4096)
    return np.apply_over_axes(np.average, extract_feature, [0])


def feature_extract_audio(model, audio):
    return model.predict(audio)


def feature_extract(post):
    # Visual Part
    visual_model = feature_visual_model()
    video = preprocessing('./' + post.content_upload.url)
    feature_visual = feature_extract_visual(visual_model, video)

    # Audio Part
    speech_model = feature_audio_model()
    mfcc = preprocess_audio(post.content_upload.url)
    feature_audio = feature_extract_audio(speech_model, mfcc)
    return np.concatenate((feature_audio, feature_visual), axis=1)


def fusion(post):
    feature = feature_extract(post)
    model = fusion_model()
    result = model.predict(feature)
    return emotions.get(np.argmax(result))
