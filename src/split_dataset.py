import os
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

emotions = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

def readFilenames():
    filenames = []
    y_labels = []
    path = "../data/ravdess_visual_frames"
    for filename in os.listdir(path):
        filenames.append(path + "/" + filename)
        y_labels.append(emotions.get(int(filename[7:8])))
    return filenames, y_labels

def writeFile(contents, filename):
    np.save(filename, contents)

def main():
    filenames, y_labels = readFilenames()
    filenames, y_labels = shuffle(filenames, y_labels)

    lb = preprocessing.LabelEncoder()
    y_labels = lb.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(filenames, y_labels, test_size=0.2, random_state=1)

    writeFile(np.array(X_train), "dataPaths/ravdess/X_train")
    writeFile(np.array(X_test), "dataPaths/ravdess/X_test")

    writeFile(np.array(y_train), "dataPaths/ravdess/Y_train")
    writeFile(np.array(y_test), "dataPaths/ravdess/Y_test")


main()
