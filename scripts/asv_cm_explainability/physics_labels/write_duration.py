import os
import sys
import glob

import soundfile as sf

def read_duration(audio_path):
    y, sr = sf.read(audio_path)
    duration = len(y) / sr
    return duration


if __name__ == "__main__":
    data_dir = sys.argv[1]

    flac_files = glob.glob(data_dir + "/trimmed/*.wav")
    with open(data_dir + "/utt2dur", "w") as w:
        for f in flac_files:
            utt = os.path.basename(f).split(".")[0]
            dur = read_duration(f)
            w.write("{} {}\n".format(utt, dur))
