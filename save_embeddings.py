from aasist.data_utils import Dataset_ASVspoof2019_devNeval
from aasist.models.AASIST import Model as AASISTModel
from ECAPATDNN.model import ECAPA_TDNN 

from torch.utils.data import DataLoader
from utils import load_parameters
from pathlib import Path
from tqdm import tqdm
import pickle as pk
import numpy as np
import torch
import json, os, argparse


SET_LIST = ['train', 'dev', 'eval']
SET_TRIAL = {
    'train': './LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
    'dev': './LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
    'eval': './LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
}
SET_DIR ={
    'train': './LA/ASVspoof2019_LA_train/',
    'dev': './LA/ASVspoof2019_LA_dev/',
    'eval': './LA/ASVspoof2019_LA_eval/'
}
SET_TRN = {
    'dev': 
    ['./LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.female.trn.txt',
    './LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.male.trn.txt'],
    'eval': 
    ['./LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt',
    './LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt']
}

def save_embeddings(set_name, cm_embd_ext, asv_embd_ext, cm_embd_path, asv_embd_path, device):
    meta_lines = open(SET_TRIAL[set_name], 'r').readlines()
    utt2spk = {}
    utt_list = []
    for line in meta_lines:
        tmp = line.strip().split(' ')
        
        spk = tmp[0]
        utt = tmp[1]

        if utt in utt2spk:
            print('Duplicated utt error', utt)
        
        utt2spk[utt] = spk
        utt_list.append(utt)

    base_dir = SET_DIR[set_name]
    dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
    loader = DataLoader(dataset,
                        batch_size=30,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=True)

    cm_emb_dic = {}
    asv_emb_dic = {}
    
    print('Getting embedgins from set %s...'%(set_name))
    
    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_cm_emb, _ = cm_embd_ext(batch_x)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()
            batch_asv_emb = asv_embd_ext(batch_x, aug = False).detach().cpu().numpy()
        
        for k, cm_emb, asv_emb in zip(key, batch_cm_emb, batch_asv_emb):
            cm_emb_dic[k] = cm_emb
            asv_emb_dic[k] = asv_emb
            
    
    with open(cm_embd_path + "%s_embeds.pk"%(set_name), "wb") as f:
        pk.dump(cm_emb_dic, f)
    with open(asv_embd_path + "%s_embeds.pk"%(set_name), "wb") as f:
        pk.dump(asv_emb_dic, f)


def save_models(set_name, asv_embd_ext, asv_embd_path, device):
    utt2spk = {}
    utt_list = []

    for trn in SET_TRN[set_name]:
        meta_lines = open(trn, 'r').readlines()
        
        for line in meta_lines:
            tmp = line.strip().split(' ')
            
            spk = tmp[0]
            utts = tmp[1].split(',')

            for utt in utts:
                if utt in utt2spk:
                    print('Duplicated utt error', utt)
            
                utt2spk[utt] = spk
                utt_list.append(utt)

    base_dir = SET_DIR[set_name]
    dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
    loader = DataLoader(dataset,
                        batch_size=30,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=True)
    asv_emb_dic = {}
    
    print('Getting embedgins from set %s...'%(set_name))
    
    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_asv_emb = asv_embd_ext(batch_x, aug = False).detach().cpu().numpy()
        
        for k, asv_emb in zip(key, batch_asv_emb):
            utt = k
            spk = utt2spk[utt]

            if spk not in asv_emb_dic:
                asv_emb_dic[spk] = []
            
            asv_emb_dic[spk].append(asv_emb)
    
    for spk in asv_emb_dic:
        asv_emb_dic[spk] = np.mean(asv_emb_dic[spk], axis = 0)

    with open(asv_embd_path + "%s_spkmodels.pk"%(set_name), "wb") as f:
        pk.dump(asv_emb_dic, f)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-meta_path", type=str, default="./data/_meta/")
    parser.add_argument("-cm_embd_path", type=str, default="./data/CM/")
    parser.add_argument("-asv_embd_path", type=str, default="./data/ASV/")
    
    parser.add_argument("-aasist_config", type=str, default="./aasist/config/AASIST.conf")
    parser.add_argument("-aasist_weight", type=str, default="./aasist/models/weights/AASIST.pth")
    parser.add_argument("-ecapa_weight", type=str, default="./ECAPATDNN/exps/pretrain.model")
    

    return parser.parse_args()

def main():
    args = get_args()
    
    if not os.path.exists(args.cm_embd_path):
        os.makedirs(args.cm_embd_path)
    if not os.path.exists(args.asv_embd_path):
        os.makedirs(args.asv_embd_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    with open(args.aasist_config, "r") as f_json:
        config = json.loads(f_json.read())

    model_config = config["model_config"]
    cm_embd_ext = AASISTModel(model_config).to(device)
    load_parameters(cm_embd_ext.state_dict(), args.aasist_weight)
    cm_embd_ext.to(device)
    cm_embd_ext.eval()
    
    asv_embd_ext = ECAPA_TDNN(C = 1024)
    load_parameters(asv_embd_ext.state_dict(), args.ecapa_weight)
    asv_embd_ext.to(device)
    asv_embd_ext.eval()

    for set_name in SET_LIST:
        save_embeddings(set_name, cm_embd_ext, asv_embd_ext, args.cm_embd_path, args.asv_embd_path, device)
        if set_name == 'train':
            continue
        save_models(set_name, asv_embd_ext, args.asv_embd_path, device)
            

if __name__ == "__main__":
    main()