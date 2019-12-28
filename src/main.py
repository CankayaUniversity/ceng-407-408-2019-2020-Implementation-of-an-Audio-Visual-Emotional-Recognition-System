import numpy as np
import os
from sklearn.utils import shuffle

from model import C3D
from generator import Generator

batch_size = 1
def generator(feat_path, label_path):
    while 1:
        x = np.load(feat_path)
        y = np.load(label_path)
        lst = range(x.shape[0])
        shuffle(lst)
        iters = int(len(lst)/batch_size)
        for i in range(iters):
            data = np.load(x[lst[(i*batch_size):((i+1)*batch_size)]][0])
            label = np.repeat(y[lst[(i*batch_size):((i+1)*batch_size)]], data.shape[0])
            #print(x[lst[(i*batch_size):((i+1)*batch_size)]][0])
            yield (data, label)


def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = C3D(16, 227, 227, 3, 8).getModel()
    for layer in model.layers[:-3]:
        layer.trainable = False
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(generator = generator("dataPaths/X_train.npy", "dataPaths/Y_train.npy"),
                        steps_per_epoch = np.load("dataPaths/X_train.npy").shape[0],
                        validation_data = generator("dataPaths/X_test.npy", "dataPaths/Y_test.npy"),
                        validation_steps = np.load("dataPaths/X_test.npy").shape[0],
                        epochs = 10)

main()
