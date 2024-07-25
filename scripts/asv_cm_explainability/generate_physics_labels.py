"""
Generate physical labels estimated from other toolkits
and write the metadata file

The generate file will be similar to 
    data/Database/ASVspoof_VCTK_aligned_meta.tsv
---------------
ASVSPOOF_ID	TAR_SPK_ID	AGE	GENDER	ACCENTS	REGION
LA_T_1097604	p276	24	F	English	Oxford
LA_T_1127197	p305	19	F	American	Philadelphia
LA_T_1167346	p247	22	M	Scottish	Argyll
LA_T_1202662	p270	21	M	English	Yorkshire
LA_T_1229885	p244	22	F	English	Manchester

Note: for emotion classifier from speechbrain, we need to install develop version of the toolkit:
    !pip install git+https://github.com/speechbrain/speechbrain.git@develop
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa


class PitchEstimator(object):
    def __init__(self):
        pass

    def estimate_pitch(self, audio_path):
        y, sr = librosa.load(audio_path)
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[np.nonzero(pitches)])
        return pitch

    def process_folder(self, folder_path):
        results = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                results[file_name] = self.estimate_pitch(file_path)
        return results


class SpeakingRate(object):
    def __init__(self):
        pass

    def estimate_speaking_rate(self, audio_path):
        y, sr = librosa.load(audio_path)
        # Example method to estimate speaking rate
        # Placeholder for actual implementation
        speaking_rate = len(librosa.effects.split(y)) / (len(y) / sr)
        return speaking_rate

    def process_folder(self, folder_path):
        results = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                results[file_name] = self.estimate_speaking_rate(file_path)
        return results


class DurationReader(object):
    def __init__(self):
        pass

    def read_duration(self, audio_path):
        y, sr = sf.read(audio_path)
        duration = len(y) / sr
        return duration

    def process_folder(self, folder_path):
        results = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                results[file_name] = self.read_duration(file_path)
        return results


class SNREstimator(object):
    def __init__(self):
        pass

    def estimate_snr(self, audio_path):
        y, sr = librosa.load(audio_path)
        # Example method to estimate SNR
        signal_power = np.mean(y**2)
        noise_power = np.mean((y - librosa.effects.hpss(y)[1]) ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def process_folder(self, folder_path):
        results = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                results[file_name] = self.estimate_snr(file_path)
        return results


if __name__ == "__main__":
    asvspoof2019_trn_wav_dir = sys.argv[1]
    asvspoof2019_dev_wav_dir = sys.argv[2]
    asvspoof2019_eval_wav_dir = sys.argv[3]

    num_roundings = 4

    # initiate the processors
    pitch_estimator = PitchEstimator()
    speaking_rate_estimator = SpeakingRate()
    duration_reader = DurationReader()
    snr_estimator = SNREstimator()

    # iterate over wav files and write the TSV file
    trn_wavs = glob.glob(asvspoof2019_trn_wav_dir + "/flac/*.flac")
    dev_wavs = glob.glob(asvspoof2019_dev_wav_dir + "/flac/*.flac")
    eval_wavs = glob.glob(asvspoof2019_eval_wav_dir + "/flac/*.flac")
    total_wavs = trn_wavs + dev_wavs + eval_wavs
    print("Num. of total wavs to handle: {}".format(len(total_wavs)))

    wav_file_names = []
    pitches = []
    speaking_rates = []
    durations = []
    snrs = []

    for wav_file in total_wavs:
        wav_file_names.append(os.path.basename(wav_file).split(".")[0])
        pitches.append(round(pitch_estimator.estimate_pitch(wav_file), num_roundings))
        speaking_rates.append(
            round(
                speaking_rate_estimator.estimate_speaking_rate(wav_file), num_roundings
            )
        )
        durations.append(round(duration_reader.read_duration(wav_file), num_roundings))
        snrs.append(round(snr_estimator.estimate_snr(wav_file), num_roundings))
        print("Processed {}".format(wav_file))

    # Prepare a dictionary for the DataFrame
    data = {
        "ASVSPOOF_ID": wav_file_names,
        "PITCH": pitches,
        "SPK_RATE": speaking_rates,
        "DURATION": durations,
        "SNR": snrs,
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a TSV file
    df.to_csv(
        "ASVspoof_VCTK_aligned_physical_meta.tsv",
        sep="\t",
        index=False,
    )
