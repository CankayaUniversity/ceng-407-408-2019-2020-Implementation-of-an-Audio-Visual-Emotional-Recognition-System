import numpy as np
from tensorflow.keras.utils import Sequence


class Generator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = []
        y = []
        for file_index in range(len(batch_x)):
            data = np.load(batch_x[file_index])
            x.append(data)
            y.extend([batch_y[file_index]] * data.shape[0])
        return np.vstack(x), np.array(y)
