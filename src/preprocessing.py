import os
import numpy as np
from progress.bar import ChargingBar

from split_frame import Frame
from video_to_frame import VideoParser

video_path = "../data/videos"

def preprocessing():
    parser = VideoParser(227, 227)
    frame = Frame(16, 227, 227, 3)
    for folders in os.listdir(video_path):
        folder = video_path + "/" + folders
        bar = ChargingBar('Folder Images Loading: ', max = len(os.listdir(folder)), suffix = '%(percent)d%%')
        for emotion in os.listdir(folder):
            video = video_path + "/" + folders + "/" + emotion
            frames = parser.readVideo(video)
            prepare = frame.getFrame(frames)
            np.save(video_path + "/" + "all" + "/" + emotion[:-4], prepare)
            bar.next()
        bar.finish()

preprocessing()