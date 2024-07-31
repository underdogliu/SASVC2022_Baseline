"""
Align physical labels estimated from the generate files
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

import numpy as np
import pandas as pd


def read_scp(in_file):
    out_dict = {}
    with open(in_file, "r") as i:
        for line in i:
            utt, value = line.split()
            out_dict[utt] = value
    return out_dict


if __name__ == "__main__":
    train_dir = sys.argv[1]
    dev_dir = sys.argv[2]
    eval_dir = sys.argv[3]
    for i in ["utt2dur", "utt2pitch", "utt2snr", "utt2spkrate"]:
        for j in [train_dir, dev_dir, eval_dir]:
            assert os.path.exists(j + "/" + i)

    utt_dur = {}
    utt_pitch = {}
    utt_snr = {}
    utt_spkrate = {}
    for data_dir in [train_dir, dev_dir, eval_dir]:
        this_utt_dur = read_scp(data_dir + "/utt2dur")
        this_utt_pitch = read_scp(data_dir + "/utt2pitch")
        this_utt_snr = read_scp(data_dir + "/utt2snr")
        this_utt_spkrate = read_scp(data_dir + "/utt2spkrate")
        utt_dur = {**utt_dur, **this_utt_dur}
        utt_pitch = {**utt_pitch, **this_utt_pitch}
        utt_snr = {**utt_snr, **this_utt_snr}
        utt_spkrate = {**utt_spkrate, **this_utt_spkrate}

    # Dictionary of dictionaries to preserve variable names
    fin_aligned = {
        "DURATION": utt_dur,
        "PITCH": utt_pitch,
        "SNR": utt_snr,
        "SPK_RATE": utt_spkrate,
    }

    # Convert dictionaries to DataFrames and merge them
    df = pd.DataFrame(fin_aligned)

    # Reset index to include the keys as a column
    df.reset_index(inplace=True)
    df.rename(columns={"index": "key"}, inplace=True)

    # Write to a TSV file
    df.to_csv("ASVspoof_VCTK_aligned_physical_meta.tsv", sep="\t", index=False)
