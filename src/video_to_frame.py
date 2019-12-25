import cv2 as cv
import numpy as np


class VideoParser:
    def __init__(self, width, height):
        self.dimension = (width, height)
        self.face_cascade = cv.CascadeClassifier('../data/haarcascade_frontalface_default.xml')

    def readVideo(self, path):
        frame_list = []
        capture = cv.VideoCapture(path)
        while(capture.isOpened()):
            ret, frame = capture.read()
            if(ret):
                faces = self.face_cascade.detectMultiScale(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), 1.3, 5)
                for (x, y, w, h) in faces:
                    frame = frame[y: y+h, x: x+w]
                    if(frame.size != 0):
                        frame = cv.resize(frame, self.dimension, interpolation=cv.INTER_AREA)
                        frame_list.append(frame)
            else:
                capture.release()
                break
        return np.array(frame_list)