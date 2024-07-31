import os
import sys
import glob

import numpy as np
import librosa


def estimate_snr(audio_path):
    y, sr = librosa.load(audio_path)
    # Example method to estimate SNR
    signal_power = np.mean(y**2)
    noise_power = np.mean((y - librosa.effects.hpss(y)[1]) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


if __name__ == "__main__":
    data_dir = sys.argv[1]

    flac_files = glob.glob(data_dir + "/trimmed/*.wav")
    with open(data_dir + "/utt2snr", "w") as w:
        for f in flac_files:
            utt = os.path.basename(f).split(".")[0]
            snr = estimate_snr(f)
            w.write("{} {}\n".format(utt, snr))
