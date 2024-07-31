# Perform silence trimming on the flac file
# Note: this simple script only trim the silences at the beginning and end of the
#   audio, not in the middle of the speech. For the latter purpose we need energy-
#   based VAD or something like that, but that might be too complicated?

import os
import sys
import glob

from shutil import copyfile


if __name__ == "__main__":

    in_wav_dir = sys.argv[1]
    out_wav_dir = sys.argv[2]
    os.makedirs(out_wav_dir + "/wav", exist_ok=True)

    flac_files = glob.glob(in_wav_dir + "/flac/*.flac")
    for in_wav_path in flac_files:
        utt = os.path.basename(f).split(".")[0]
        out_wav_path = out_wav_dir + "/{}.wav".format(utt)

        cmd = "sox {0} file_temp.flac silence 1 0.1 0.5% reverse && sox file_temp.flac {1} silence 1 0.1 0.5% reverse && rm file_temp.flac".format(
            in_wav_path, out_wav_path
        )
        os.system(cmd)
