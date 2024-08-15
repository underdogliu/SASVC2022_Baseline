import os
import sys

import glob


if __name__ == "__main__":
    data_dir = sys.argv[1]
    wav_files = glob.glob(data_dir + "/trimmed/*.wav")

    with open(data_dir + "/wav.scp", "w") as w:
        for wav_path in wav_files:
            utt = os.path.basename(wav_path).split(".")[0]
            w.write("{} {}\n".format(utt, wav_path))
