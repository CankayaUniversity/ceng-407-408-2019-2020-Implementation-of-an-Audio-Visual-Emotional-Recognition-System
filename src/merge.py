import pandas as pd

pd.set_option('display.expand_frame_repr', False)


def merge(df, df1):
    return pd.merge(df, df1, on='Filename')


m = merge(pd.read_hdf("audio.h5"), pd.read_hdf("visual.h5").drop(['Emotion'], axis=1))
print(m.head())
m.to_hdf("audio_visual_feature.h5", key="features")
print(m.shape)
