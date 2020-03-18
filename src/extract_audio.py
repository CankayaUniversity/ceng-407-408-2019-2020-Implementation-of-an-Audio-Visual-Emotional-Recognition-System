import glob
import librosa
import numpy as np
import joblib

data = glob.glob("../data/ravdess_extract_audio/*.wav")
lst = []

for audio in data:
    try:
        X, sample_rate = librosa.load(audio, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        filename = audio[-24:-4]
        file = int(filename[7:8]) - 1
        if filename[1] != '2':
            arr = mfccs, file
            lst.append(arr)
    except ValueError:
        continue

X, y = zip(*lst)
X = np.asarray(X)
y = np.asarray(y)
print(X.shape, y.shape)

X_name = 'X.joblib'
y_name = 'y.joblib'
joblib.dump(X, X_name)
joblib.dump(y, y_name)
