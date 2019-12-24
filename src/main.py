import numpy as np
import os

from model import C3D


def batch_generator(X, Y, batch_size):
    indices = np.arange(len(X))
    batchs = []
    while True:
        np.random.shuffle(indices)
        for i in indices:
            batchs.append(i)
            if(len(batchs) == batch_size):
                data = []
                label = []
                for batch in batchs:
                    value = np.load(X[batch])
                    data.append(value)
                    label.extend([Y[batch]] * value.shape[0])
                yield np.vstack(data), np.vstack(label)
                batch = []
                data = []
                label = []


def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    batch_size = 1
    train = batch_generator(np.load("dataPaths/X_train.npy"), np.load("dataPaths/Y_train.npy"), batch_size)
    test = batch_generator(np.load("dataPaths/X_test.npy"), np.load("dataPaths/Y_test.npy"), batch_size)

    model = C3D(16, 227, 227, 3, 8).getModel()
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model.fit_generator(generator=train,
                        steps_per_epoch=int(np.load("dataPaths/X_train.npy").shape[0] // batch_size),
                        epochs=10,
                        validation_data=test,
                        validation_steps=int(np.load("dataPaths/X_test.npy").shape[0] // batch_size))


main()
