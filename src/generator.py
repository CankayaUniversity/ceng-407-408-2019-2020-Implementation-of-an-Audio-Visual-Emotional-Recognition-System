import numpy as np
from sklearn.utils import shuffle

class Generator:
  def __init__(self, feat_path, label_path, batch_size):
    self.feat_path = feat_path
    self.label_path = label_path
    self.video_size = 1
    self.batch_size = batch_size

  def pick(self):
      batch_data = []
      batch_label = []
      flag = 1
      while 1:
          x = np.load(self.feat_path)
          y = np.load(self.label_path)
          lst = range(x.shape[0])
          shuffle(lst)
          iters = int(len(lst) / self.video_size)
          for i in range(iters):
              data = np.load(x[lst[(i * self.video_size):((i + 1) * self.video_size)]][0])
              label = y[lst[(i * self.video_size):((i + 1) * self.video_size)]]
              for index in range(data.shape[0]):
                  batch_data.append(data[index])
                  batch_label.append(label)
                  if (flag % self.batch_size == 0):
                      #print(np.array(batch_data).shape, np.array(batch_label).shape)
                      yield (np.array(batch_data), np.array(batch_label))
                      batch_data = []
                      batch_label = []
                      flag = 0
                  flag = flag + 1
          if (flag > 1):
            yield (np.array(batch_data), np.array(batch_label))
            batch_data = []
            batch_label = []
            flag = 0