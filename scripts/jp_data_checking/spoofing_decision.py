"""
Instead of evaluation, outputting the decision of bonafide/spoof
with the input of a wav file
"""

import argparse
import json
import os
import pickle as pk
from pathlib import Path

import numpy as np
import librosa
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from aasist.models.AASIST import Model as AASISTModel
from ECAPATDNN.model import ECAPA_TDNN
from utils import load_parameters


class Dataset_ASVspoof2019_Varylength(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),"""
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = librosa.read(str(self.base_dir / f"wavs/{key}.wav"))
        x_inp = Tensor(X)
        return x_inp, key


# list of dataset partitions
SET_PARTITION = ["trn", "dev", "eval"]

# list of countermeasure(CM) protocols
SET_CM_PROTOCOL = {
    "trn": "protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

# directories of each dataset partition
SET_DIR = {
    "trn": "./LA/ASVspoof2019_LA_train/",
    "dev": "./LA/ASVspoof2019_LA_dev/",
    "eval": "./LA/ASVspoof2019_LA_eval/",
}

# enrolment data list for speaker model calculation
# each speaker model comprises multiple enrolment utterances
SET_TRN = {
    "dev": [
        "./LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.female.trn.txt",
        "./LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.male.trn.txt",
    ],
    "eval": [
        "./LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt",
        "./LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt",
    ],
}


def save_decisions(input_wav_list, cm_embd_ext, device):
    wav_list = []
    with open(input_wav_list, "r") as w:
        for line in w:
            wav_path = line.strip()
            wav_list.append(wav_path)
    base_dir = os.path.dirname(input_wav_list)
    dataset = Dataset_ASVspoof2019_Varylength(wav_list, Path(base_dir))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True
    )

    with open(base_dir + "/out_decision.txt", "w") as o:
        for batch_x, key in tqdm(loader):
            batch_x = batch_x.to(device)
            with torch.no_grad():
                _, decision = cm_embd_ext(batch_x)
                o.write("{} {}\n".format(key, decision))
                if decision == 1:
                    print("{} looks synthetic to the model".format(key))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-aasist_config", type=str, default="./aasist/config/AASIST.conf"
    )
    parser.add_argument(
        "-aasist_weight", type=str, default="./aasist/models/weights/AASIST.pth"
    )
    parser.add_argument("-input_wav_list", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    with open(args.aasist_config, "r") as f_json:
        config = json.loads(f_json.read())

    model_config = config["model_config"]
    cm_embd_ext = AASISTModel(model_config).to(device)
    load_parameters(cm_embd_ext.state_dict(), args.aasist_weight)
    cm_embd_ext.to(device)
    cm_embd_ext.eval()

    save_decisions(args.input_wav_list, cm_embd_ext, device)


if __name__ == "__main__":
    main()
