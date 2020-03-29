import os
import numpy as np
from progress.bar import ChargingBar

from split_frame import Frame
from video_to_frame import VideoParser


class Preprocessing:
    def __init__(self):
        self.parser = VideoParser(227, 227)
        self.frame = Frame(16, 227, 227, 3)

    def preprocessing(self, video_path, folderName, video_filename):
        id = 1
        for folders in os.listdir(video_path):
            folder = video_path + "/" + folders
            bar = ChargingBar('{} Folder Images Loading: '.format(folders), max=len(os.listdir(folder)),
                              suffix='%(percent)d%%')
            for emotion in os.listdir(folder):
                video = video_path + "/" + folders + "/" + emotion
                frames = self.parser.readVideo(video)
                prepare = self.frame.getFrame(frames)
                np.save("../data/" + folderName + "/" + emotion[:video_filename] + str(id), prepare)
                bar.next()
                id = id + 1
        bar.finish()
