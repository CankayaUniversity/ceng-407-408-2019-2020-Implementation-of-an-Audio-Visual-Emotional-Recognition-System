import numpy as np

from generator import Generator
from model import C3D

def main():
    model = C3D(16, 227, 227, 3, 8).getModel()
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    batch_size = 16
    train = Generator(np.load("X_train.npy"), np.load("Y_train.npy"), batch_size)
    test = Generator(np.load("X_test.npy"), np.load("Y_test.npy"), batch_size)
    
    model.fit_generator(generator=train,
                   steps_per_epoch = int(np.load("X_train.npy").shape[0] // batch_size),
                   epochs = 10,
                   verbose = 1,
                   validation_data = test,
                   validation_steps = int(np.load("X_test.npy").shape[0] // batch_size))
main()