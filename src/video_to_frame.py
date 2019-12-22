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
                    img = cv.resize(frame[y : y+h, x : x+w], self.dimension, interpolation = cv.INTER_AREA)
            else:
                capture.release()
                break
            frame_list.append(img)
        return np.array(frame_list)