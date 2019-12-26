import numpy as np
import os

from model import C3D
from generator import Generator


def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    batch_size = 1
    model = C3D(16, 227, 227, 3, 8).getModel()
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(generator(np.load("dataPaths/X_train.npy"), np.load("dataPaths/Y_train.npy"), batch_size), 
                    steps_per_epoch = int(np.load("dataPaths/X_train.npy").shape[0] // batch_size),
                    validation_data = generator(np.load("dataPaths/X_test.npy"), np.load("dataPaths/Y_test.npy"), batch_size),
                    validation_steps = int(np.load("dataPaths/X_test.npy").shape[0] // batch_size),
                    epochs = 10)

main()
