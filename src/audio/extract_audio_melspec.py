import glob
import numpy as np
import librosa
from skvideo import io
from moviepy.editor import *


def melspecs(audio_file):
    waveform, sample_rate = librosa.load(audio_file)
    filename1 = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=64, fmin=20, fmax=11000,
                                               window='hamming', win_length=275)

    delta1 = librosa.feature.delta(filename1, order=1)
    delta2 = librosa.feature.delta(filename1, order=2)
    audio_feature = np.dstack((filename1, delta1, delta2))

    return np.resize(audio_feature, (227, 227, 3))


def audioVisualSplit(path):
    return VideoFileClip(path).audio, io.vread(path)


def start(path):
    audio, _ = audioVisualSplit(path)
    audio.write_audiofile("audio.wav", logger=None)
    feature = melspecs("audio.wav")
    np.save("../data/ravdess_audio_frames_melspec/" + path[-24:-4], feature)
    print(path[-24:-4], feature.shape)


ok_list = glob.glob("../data/ravdess_audio_frames_melspec/*.npy")
l = []
for index in range(len(ok_list)):
    l.append(ok_list[index][-24:-4])

file_path = "../data/ravdess_videos/Actor_*/*.mp4"

for file in glob.glob(file_path):
    if not file[-24:-4] in l:
        start(file)
