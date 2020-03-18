import pandas as pd
import numpy as np
from sklearn.utils import shuffle

pd.set_option('display.expand_frame_repr', False)
df = pd.read_hdf("audio_visual_feature.h5")

data = []
for i in range(df.shape[0]):
    try:
        concat = np.concatenate((df['Audio Feature'][i], df['Visual Feature'][i]), axis=1)
        data.append([concat, df['Emotion'][i]])
    except:
        continue

df2 = shuffle(pd.DataFrame(data=data, columns=['Features', 'Emotion']))
print(df2.head(10))
df2.to_hdf("features.h5", key="data")
print(df2.shape)
