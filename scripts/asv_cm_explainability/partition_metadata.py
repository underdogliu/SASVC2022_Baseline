
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the file into a DataFrame
file_path = 'data/Database/ASVspoof_VCTK_aligned_meta_partitioned_eval_bonafide.tsv'
df = pd.read_csv(file_path, sep='\t')

# ======= Overlap Case ======= #
# Ensure all TAR_SPK_IDs are present in both partitions
train_overlap = pd.DataFrame()
eval_overlap = pd.DataFrame()

for tar_spk_id, group in df.groupby('TAR_SPK_ID'):
    train_group, eval_group = train_test_split(group, test_size=0.1, random_state=42)
    train_overlap = pd.concat([train_overlap, train_group])
    eval_overlap = pd.concat([eval_overlap, eval_group])

# Assign the PARTITION column
train_overlap['PARTITION'] = 'train'
eval_overlap['PARTITION'] = 'eval'

# Combine train and eval for the overlap case
both_partitions_df = pd.concat([train_overlap, eval_overlap])
both_partitions_df.to_csv('repartitioned_with_overlap.tsv', sep='\t', index=False)

# ======= No Overlap Case ======= #
# Split TAR_SPK_IDs into 90% train, 10% eval without overlap
tar_spk_ids = df['TAR_SPK_ID'].unique()
train_ids, eval_ids = train_test_split(tar_spk_ids, test_size=0.1, random_state=42)

train_no_overlap = df[df['TAR_SPK_ID'].isin(train_ids)].copy()
eval_no_overlap = df[df['TAR_SPK_ID'].isin(eval_ids)].copy()

# Assign the PARTITION column
train_no_overlap['PARTITION'] = 'train'
eval_no_overlap['PARTITION'] = 'eval'

# Combine train and eval for the no overlap case
no_overlap_df = pd.concat([train_no_overlap, eval_no_overlap])
no_overlap_df.to_csv('repartitioned_no_overlap.tsv', sep='\t', index=False)

print("Repartitioning complete. Files 'repartitioned_with_overlap.tsv' and 'repartitioned_no_overlap.tsv' have been saved.")

