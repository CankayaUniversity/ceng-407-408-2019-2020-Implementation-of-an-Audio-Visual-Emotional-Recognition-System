import glob
import numpy as np
import librosa
from skvideo import io
from moviepy.editor import *


def mfcc(audio_file):
    X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs.reshape((1, 40))


def audioVisualSplit(path):
    return VideoFileClip(path).audio, io.vread(path)


def start(path):
    audio, _ = audioVisualSplit(path)
    audio.write_audiofile("audio.wav", logger=None)
    feature = mfcc("audio.wav")
    np.save("../data/ravdess_audio_frames_mfcc/" + path[-24:-4], feature)
    print(path[-24:-4], feature.shape)


ok_list = glob.glob("../data/ravdess_audio_frames_mfcc/*.npy")
l = []
for index in range(len(ok_list)):
    l.append(ok_list[index][-24:-4])

file_path = "../data/ravdess_videos/Actor_*/*.mp4"

for file in glob.glob(file_path):
    if not file[-24:-4] in l:
        start(file)
