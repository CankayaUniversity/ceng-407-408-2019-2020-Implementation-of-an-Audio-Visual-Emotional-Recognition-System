
from preprocessing import Preprocessing

def preprocessing():
    pre = Preprocessing()
    pre.preprocessing("../data/ravdess_videos", "rrr_frames", -4)

def main():
    preprocessing()

main()