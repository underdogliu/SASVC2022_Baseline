"""
Prepare the VCTk and ASVspoof label files for classification task

This script prepares the statistical attributes as labels for classification
post-hoc analysis
"""

import torch
import pandas as pd

if __name__ == "__main__":
    asvspoof_vctk_meta_file = (
        "data/Database/ASVspoof_VCTK_VCC_MetaInfo/ASVspoof2019_LA_VCTK_MetaInfo.tsv"
    )
    vctk_spk_meta_file = "data/Database/CSTR_VCTK-Corpus-0.9/speaker-info.txt"
    vctk_text_trans_dir = "data/Database/CSTR_VCTK-Corpus-0.9/txt"

    asvspoof_utt2emo_meta = {}
    with open("data/Database/ASVspoof_VCTK_VCC_MetaInfo/utt2emo", "r") as u:
        asvspoof_utt2emo_meta[u.split()[0]] = u.split()[1]

    # Read the mapping meta
    asvspoof_vctk_mapping = {}
    with open(asvspoof_vctk_meta_file, "r") as a:
        for line in a:
            asvspoof_id, vctk_id, target_speaker_id, tts_text, _ = line.split("\t")
            if target_speaker_id == "-" and "_" in vctk_id:
                target_speaker_id = vctk_id.split("_")[0]

            asvspoof_vctk_mapping[asvspoof_id] = [vctk_id, target_speaker_id, tts_text]

    # Read VCTK speaker-wise data
    vctk_spk_meta = {}
    with open(vctk_spk_meta_file, "r") as v:
        for line in v:
            line_sp = line.split()
            speaker_id, age, gender, accents = line_sp[0:4]
            region = "_".join(line_sp[4:])
            speaker_id = "p" + speaker_id
            vctk_spk_meta[speaker_id] = {
                "age": age,
                "gender": gender,
                "accents": accents,
                "region": region,
            }

    # write large aligned file
    # ASVSpoof ID | TAR SPK ID | AGE | GENDER | ACCENTS | REGION
    # Prepare data for DataFrame
    data = []
    for asvspoof_id, (
        vctk_id,
        target_speaker_id,
        tts_text,
    ) in asvspoof_vctk_mapping.items():
        if target_speaker_id in vctk_spk_meta.keys():
            speaker_meta = vctk_spk_meta[target_speaker_id]
            row = {
                "ASVSPOOF_ID": asvspoof_id,
                "TAR_SPK_ID": target_speaker_id,
                "AGE": speaker_meta["age"],
                "GENDER": speaker_meta["gender"],
                "ACCENTS": speaker_meta["accents"],
                "REGION": speaker_meta["region"],
                "EMOTION": asvspoof_utt2emo_meta[asvspoof_id],
            }
            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Write to TSV file
    df.to_csv("data/Database/ASVspoof_VCTK_aligned_meta.tsv", sep="\t", index=False)
