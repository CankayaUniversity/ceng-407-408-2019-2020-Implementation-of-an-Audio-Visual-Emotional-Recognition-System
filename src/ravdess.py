import os
import numpy as np

from models import C3D
from generator import Generator

# -----Total----
# 2880  -> video
# 21161 -> segment

# -----Train----
# 2304  -> video
# 16919 -> segment

# -----Test-----
# 576 -> video
# 4242 -> segment

def main():
    batch_size = 1

    model = C3D(16, 227, 227, 3, 8).getModel(False)
    model.compile(optimizer="adadelta", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    train = Generator("dataPaths/ravdess/X_train.npy", "dataPaths/ravdess/Y_train.npy", batch_size)
    test = Generator("dataPaths/ravdess/X_test.npy", "dataPaths/ravdess/Y_test.npy", batch_size)
    
    model.fit_generator(generator = train.pick(),
                        steps_per_epoch = int(16919/batch_size),
                        validation_data = test.pick(),
                        validation_steps = int(4242/batch_size),
                        epochs = 10)

    ## Test
    test_index = np.load("dataPaths/ravdess/X_test.npy")[10]
    prd = model.predict(np.load(test_index))
    print("Real Emotion: ", np.load("dataPaths/ravdess/Y_test.npy")[10])
    print("Predict Emotion for each segment: ", np.argmax(prd, axis = 1))

main()