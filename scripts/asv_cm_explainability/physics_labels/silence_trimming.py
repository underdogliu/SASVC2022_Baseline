# Perform silence trimming on the flac file
# Note: this simple script only trim the silences at the beginning and end of the
#   audio, not in the middle of the speech. For the latter purpose we need energy-
#   based VAD or something like that, but that might be too complicated?

import os
import sys

from shutil import copyfile


if __name__ == "__main__":

    in_wav_dir = sys.argv[1]
    out_wav_dir = sys.argv[2]
    os.makedirs(out_wav_dir + "/wav", exist_ok=True)

    for f in ["wav.scp", "utt2spk"]:
        assert os.path.exists(in_wav_dir + "/{}".format(f))

    with open(in_wav_dir + "/wav.scp", "r") as w, open(
        out_wav_dir + "/wav.scp", "w"
    ) as o:
        for line in w:
            utt, in_wav_path = line.split()

            out_wav_path = out_wav_dir + "/wav/{}".format(os.path.basename(in_wav_path))
            if not os.path.exists(os.path.dirname(out_wav_path)):
                os.makedirs(os.path.dirname(out_wav_path))

            cmd = "sox {0} file_temp.flac silence 1 0.1 0.5% reverse && sox file_temp.flac {1} silence 1 0.1 0.5% reverse && rm file_temp.flac".format(
                in_wav_path, out_wav_path
            )
            os.system(cmd)

            o.write("{} {}\n".format(utt, out_wav_path))

    copyfile(in_wav_dir + "/utt2spk", out_wav_dir + "/utt2spk")
