from split_frame import Frame
from video_to_frame import VideoParser
from model import C3D

def createModel():
    return C3D(16, 227, 227, 3, 8).getModel()

def preprocessing():
    parser = VideoParser(227, 227)
    frames = parser.readVideo("../data/videos/video1.mp4")
    prepare = Frame(16, 227, 227, 3).frame(frames)
    print(prepare.shape)

model = createModel()
print(model.summary())
preprocessing()