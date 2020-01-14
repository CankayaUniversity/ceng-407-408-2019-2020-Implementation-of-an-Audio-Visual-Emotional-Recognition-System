import os
import numpy as np

from models import C3D
from generator import Generator

# -----Train----
# 2304  -> video
# 16927 -> frames

# -----Test-----
# 576 -> video
# 4234 -> frames

def main():
    batch_size = 1

    model = C3D(16, 227, 227, 3, 8).getModel(False)
    model.compile(optimizer="adadelta", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    train = Generator("dataPaths/X_train.npy", "dataPaths/Y_train.npy", batch_size)
    test = Generator("dataPaths/X_test.npy", "dataPaths/Y_test.npy", batch_size)
    
    model.fit_generator(generator = train.pick(),
                        steps_per_epoch = int(16927/batch_size),
                        validation_data = test.pick(),
                        validation_steps = int(4234/batch_size),
                        epochs = 10)

    ## Test
    test_index = np.load("dataPaths/X_test.npy")[10]
    prd = model.predict(np.load(test_index))
    print("Real Emotion: ", np.load("dataPaths/Y_test.npy")[10])
    print("Predict Emotion for each segment: ", np.argmax(prd, axis = 1))

main()