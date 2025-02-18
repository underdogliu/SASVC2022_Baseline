import random

# Dictionary mapping
attack_dict = {
    "A18": {"group": 1, "type": "VC", "am": "ivector+plda", 'wm': 'lpc'},
    "A06": {"group": 1, "type": "VC", "am": "GMM", 'wm': 'spectral_filtering'},
    "A19": {"group": 1, "type": "VC", "am": "GMM", 'wm': 'spectral_filtering'},
    "A05": {"group": 1, "type": "VC", "am": "VAE", 'wm': 'world'},
    "A17": {"group": 1, "type": "VC", "am": "VAE", 'wm': 'waveform_filtering'},
    "A13": {"group": 2, "type": "TTS", "am": "TTSVC(DNN)", 'wm': 'spectral_filtering'},
    "A09": {"group": 2, "type": "TTS", "am": "NLP+RNN", 'wm': 'vocaine'},
    "A14": {"group": 2, "type": "TTS", "am": "TTSVC(DNN)", 'wm': 'STRAIGHT'},
    "A03": {"group": 2, "type": "TTS", "am": "NLP+DNN", 'wm': 'WORLD'},
    "A02": {"group": 2, "type": "TTS", "am": "NLP+HMM+DNN", 'wm': 'WORLD'},
    "A07": {"group": 2, "type": "TTS", "am": "NLP+RN+GAN", 'wm': 'WORLD'},
    "A11": {"group": 2, "type": "TTS", "am": "end2end", 'wm': 'GriffinLim'},
    "A08": {"group": 3, "type": "TTS", "am": "NLP+HMM+DNN", 'wm': 'DilatedCNN'},
    "A01": {"group": 3, "type": "TTS", "am": "NLP+HMM+DNN", 'wm': 'WaveNet'},
    "A12": {"group": 3, "type": "TTS", "am": "NLP+RNN", 'wm': 'WaveNet'},
    "A15": {"group": 3, "type": "TTS", "am": "TTS+VC(DNN)", 'wm': 'WaveNet'},
    "A10": {"group": 3, "type": "TTS", "am": "end2end", 'wm': 'WaveRNN'},
    "A04": {"group": 4, "type": "TTS", "am": "NLP+unit-selection", 'wm': 'wave.concat.'},
    "A06": {"group": 4, "type": "TTS", "am": "NLP+unit-selection", 'wm': 'wave.concat.'},
    "A16": {"group": 4, "type": "TTS", "am": "CART", 'wm': 'wave.concat.'}
}

# Input and output file paths
input_file = "../synthetiq-audio-io-board/data/asvspoof2019/LA/eval/trimmed_norm/asvspoof2019_trials.txt"
output_file = "ASVspoof_VCTK_aligned_meta_partitioned_eval_cm_class.tsv"

# Reading the input metadata file
with open(input_file, 'r') as infile:
    lines = infile.readlines()

parsed_data = []

# Processing each line
for line in lines:
    parts = line.strip().split()
    asvspoof_id = parts[1]
    attack_code = parts[3]
    
    if attack_code.startswith("A"):  # Spoof case
        attack_type = attack_dict[attack_code]["type"]
        am = attack_dict[attack_code]["am"]
        wm = attack_dict[attack_code]["wm"]
    else:  # Bonafide case
        attack_type = "bonafide"
        am = "-"
        wm = "-"
    
    parsed_data.append([asvspoof_id, attack_code, attack_type, am, wm])

# Splitting into train and eval sets (90-10 split)
random.shuffle(parsed_data)

train_set = []
eval_set = []
coverage = {
    "type": set(),
    "am": set(),
    "wm": set(),
}

# Ensure at least one of each value is in both partitions
for data in parsed_data:
    if (len(eval_set) < len(parsed_data) * 0.1) or (
        data[2] not in coverage["type"] or 
        data[3] not in coverage["am"] or 
        data[4] not in coverage["wm"]
    ):
        eval_set.append(data + ["eval"])
        coverage["type"].add(data[2])
        coverage["am"].add(data[3])
        coverage["wm"].add(data[4])
    else:
        train_set.append(data + ["train"])

# Balance the train and eval partitions
remaining_for_eval = len(parsed_data) * 0.1 - len(eval_set)
if remaining_for_eval > 0:
    eval_set.extend(train_set[:int(remaining_for_eval)])
    train_set = train_set[int(remaining_for_eval):]

# Writing the output to a TSV file
with open(output_file, 'w') as outfile:
    outfile.write("ASVSPOOF_ID\tATTACK\tTYPE\tAM\tWM\tPARTITION\n")
    for data in train_set + eval_set:
        outfile.write("\t".join(data) + "\n")

print(f"TSV file '{output_file}' generated successfully.")

