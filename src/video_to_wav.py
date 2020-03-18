import subprocess
import glob

data = glob.glob("../data/ravdess_videos/Actor_*/*.mp4")
command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn ../data/ravdess_extract_audio/{}.wav"
index = 0

for audio in data:
    try:
        process = command.format(audio, data[index][-24:-4])
        index += 1
        subprocess.call(process, shell=True)
    except ValueError:
        continue
