from preprocessing import Preprocessing


def preprocessing():
    pre = Preprocessing()
    pre.preprocessing("../data/savee_videos", "savee_frames", -4)


def main():
    preprocessing()


main()
