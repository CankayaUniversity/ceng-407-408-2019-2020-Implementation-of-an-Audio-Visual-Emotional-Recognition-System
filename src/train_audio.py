import numpy as np
import joblib

from tensorflow.keras.layers import Conv1D, Activation, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow.keras.backend as K


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def get_model():
    model = Sequential()

    model.add(Conv1D(128, 5, padding='same', input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(8))
    model.add(Activation('softmax'))

    return model


def checkpoint():
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return callbacks_list


if __name__ == '__main__':
    x = joblib.load("../src/X.joblib")
    y = joblib.load("../src/y.joblib")
    x = np.expand_dims(x.reshape(x.shape[0], 40), axis=2)
    print(x.shape, y.shape)

    model = get_model()
    model.load_weights("../src/Conv1d_weights.hdf5")
    print(model.summary())
    callbacks_list = checkpoint()

    model.compile(optimizer=RMSprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy', f1_metric])
    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(x=x, y=y, epochs=1000, batch_size=32, validation_split=0.2, verbose=2,
              callbacks=[tensor_board])
