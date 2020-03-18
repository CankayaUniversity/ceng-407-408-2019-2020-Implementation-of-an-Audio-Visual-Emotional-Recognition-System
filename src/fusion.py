import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def get_model():
    model = Sequential([
        Dense(4096, input_shape=(8192,)),
        Activation('relu'),
        Dropout(0.2),

        Dense(2048),
        Activation('relu'),
        Dropout(0.2),

        Dense(1096),
        Activation('relu'),
        Dropout(0.2),

        Dense(8),
        Activation('softmax')
    ])
    return model


def checkpoint():
    filepath = "adam(0.001).100.32.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return callbacks_list


if __name__ == '__main__':
    x = np.load("X_features.npy")
    y = np.load("Y_features.npy")

    print("X: {}".format(x.shape))
    print("Y: {}".format(y.shape))

    model = get_model()
    print(model.summary())
    callbacks_list = checkpoint()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x, y=y, epochs=100, batch_size=32, verbose=2, validation_split=0.2, callbacks=callbacks_list)
