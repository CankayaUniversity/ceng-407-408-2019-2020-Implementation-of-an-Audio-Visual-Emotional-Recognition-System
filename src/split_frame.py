import numpy as np

# To make sure each video segment has 16 frames,
# i.e., the input size in C3D-Sports-1M model [33],
# we delete the ﬁrst and last L−16 2 overlapping frames
# if a video segment has L ≥ 16 frames. On the contrary,
# for L < 16 we repeat the ﬁrst and last 16−L 2
# overlapping frames.

class Frame:
    def __init__(self, frame_size, width, height, dimension):
        self.frame_size = frame_size
        self.width = width
        self.height = height
        self.dimension = dimension

    def getFrame(self, data):
        data = np.array(data)
        if data.shape[0] % self.frame_size == 0:
            final = data.reshape(int(data.shape[0] / self.frame_size), self.frame_size, self.width, self.height,
                                 self.dimension)
        else:
            overlap = (data.shape[0] - self.frame_size * int(data.shape[0] / self.frame_size))  # remain frame size / 16
            beginning = int((self.frame_size - overlap) / 2)  # (16 - L) / 2
            ending = self.frame_size - overlap - beginning
            final = np.concatenate((data[:-overlap], np.concatenate(
                (data[:beginning], np.concatenate((data[-overlap:], data[-ending:]))))))
            final = final.reshape(int(final.shape[0] / self.frame_size), self.frame_size, self.width, self.height,
                                  self.dimension)
        return final
