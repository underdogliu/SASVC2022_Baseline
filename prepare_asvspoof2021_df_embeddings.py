import argparse
import json
import os
import sys
import pickle as pk
from pathlib import Path

import glob
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aasist.data_utils import pad, pad_random
from aasist.models.AASIST import Model as AASISTModel
from ECAPATDNN.model import ECAPA_TDNN
from utils import load_parameters

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


def save_embeddings(set_name, cm_embd_ext, asv_embd_ext, device):
    meta_lines = open(SET_CM_PROTOCOL[set_name], "r").readlines()
    utt2spk = {}
    utt_list = []
    for line in meta_lines:
        tmp = line.strip().split(" ")

        spk = tmp[0]
        utt = tmp[1]

        if utt in utt2spk:
            print("Duplicated utt error", utt)

        utt2spk[utt] = spk
        utt_list.append(utt)

    base_dir = SET_DIR[set_name]
    dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
    loader = DataLoader(
        dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True
    )

    os.makedirs("npy_embeddings/{0}".format(set_name), exist_ok=True)

    print("Getting embeddings from set %s..." % (set_name))

    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_cm_emb, _ = cm_embd_ext(batch_x)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()
            batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

        for k, cm_emb, _ in zip(key, batch_cm_emb, batch_asv_emb):
            npy_file_path = "npy_embeddings/{0}/{1}".format(set_name, k)
            np.save(npy_file_path, cm_emb)


def save_embeddings_npys(src_ext_folder, tar_ext_folder, cm_embd_ext, device):
    os.makedirs(tar_ext_folder + "/npys", exist_ok=True)

    # wav_scp = src_ext_folder + "/wav.scp"
    wav_files = glob.glob(src_ext_folder + "/*.flac")

    for wav_file in wav_files:
        utt = os.path.basename(wav_file).split(".")[0]
        x, _ = sf.read(wav_file)
        x = pad_random(x)
        x = torch.Tensor(x)
        x = x.to(device)
        x = x.unsqueeze(0)
        with torch.no_grad():
            cm_emb, _ = cm_embd_ext(x)
            cm_emb = cm_emb.detach().cpu().numpy()
            cm_emb = cm_emb.squeeze()
            npy_file_path = tar_ext_folder + "/npys/{0}.npy".format(utt)
            np.save(npy_file_path, cm_emb)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-aasist_config", type=str, default="./aasist/config/AASIST.conf"
    )
    parser.add_argument(
        "-aasist_weight", type=str, default="./aasist/models/weights/AASIST.pth"
    )
    parser.add_argument(
        "-ecapa_weight", type=str, default="./ECAPATDNN/exps/pretrain.model"
    )

    return parser.parse_args()


def main():
    aasist_config = "./aasist/config/AASIST.conf"
    aasist_weight = "./aasist/models/weights/AASIST.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    with open(aasist_config, "r") as f_json:
        config = json.loads(f_json.read())

    model_config = config["model_config"]
    cm_embd_ext = AASISTModel(model_config).to(device)
    load_parameters(cm_embd_ext.state_dict(), aasist_weight)
    cm_embd_ext.to(device)
    cm_embd_ext.eval()

    srcdir = sys.argv[1]
    tardir = sys.argv[2]
    save_embeddings_npys(srcdir, tardir, cm_embd_ext, device)


if __name__ == "__main__":
    main()
