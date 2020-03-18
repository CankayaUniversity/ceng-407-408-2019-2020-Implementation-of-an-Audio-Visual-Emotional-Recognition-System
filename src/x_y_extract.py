import pandas as pd
import numpy as np

pd.set_option('display.expand_frame_repr', False)

visual = pd.read_hdf("visual.h5")
print(visual.head())
print(visual.shape)

audio = pd.read_hdf("audio.h5")
print(audio.head())
print(audio.shape)

audio_visual = pd.read_hdf("audio_visual_feature.h5")
print(audio_visual.head())
print(audio_visual.shape)

feature = pd.read_hdf("features.h5")
print(feature.head())
print(feature.shape)
print(feature['Features'][0].shape)

x = []
y = []
for i in range(feature.shape[0]):
    x.append(feature['Features'][i])
    y.append(np.array([feature['Emotion'][i]]))

x = np.array(x)
y = np.array(y)

x = x.reshape(x.shape[0], 8192)
y = y.reshape(y.shape[0], 8)

print(x.shape, y.shape)
np.save("X_features", x)
np.save("Y_features", y)
