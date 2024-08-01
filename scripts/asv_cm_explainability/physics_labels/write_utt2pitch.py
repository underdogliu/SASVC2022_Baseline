"""
Average the pitch features and store them into the numpy tarball
"""

import glob
import os
import sys

import parselmouth


def extract_pitch(audio_path):
    # Load the audio file
    snd = parselmouth.Sound(audio_path)
    # Extract pitch using Parselmouth
    pitch = snd.to_pitch()
    # Get pitch values in Hz
    pitch_values = pitch.selected_array["frequency"]
    # Remove unvoiced pitch values (represented by 0 Hz)
    pitch_values = pitch_values[pitch_values > 0]

    return pitch_values.mean()


if __name__ == "__main__":
    data_dir = sys.argv[1]

    flac_files = glob.glob(data_dir + "/trimmed/*.wav")
    with open(data_dir + "/utt2pitch", "w") as w:
        for f in flac_files:
            utt = os.path.basename(f).split(".")[0]
            dur = extract_pitch(f)
            print("pitch energy: {}".format(dur))
            w.write("{} {}\n".format(utt, dur))
