import pandas as pd
from sklearn.model_selection import train_test_split

# Load the file into a DataFrame
src_file_path = (
    "data/Database/ASVspoof_VCTK_aligned_physical_meta_partitioned_eval_bonafide.tsv"
)
df = pd.read_csv(src_file_path, sep="\t")

overlap_file_path = "data/Database/ASVspoof_VCTK_aligned_physical_meta_partitioned_eval_bonafide_no_overlap.tsv"
nooverlap_file_path = "data/Database/ASVspoof_VCTK_aligned_physical_meta_partitioned_eval_bonafide_overlap.tsv"

# Split into train and eval partitions (90-10 ratio)
train_df, eval_df = train_test_split(
    df, test_size=0.1, stratify=df["TAR_SPK_ID"], random_state=42
)

# Add a new PARTITION column
train_df["PARTITION"] = "train"
eval_df["PARTITION"] = "eval"

# Combine train and eval DataFrames
combined_df = pd.concat([train_df, eval_df])

# Ensure all TAR_SPK_IDs are present in both train and eval
tar_spk_ids = df["TAR_SPK_ID"].unique()
for spk_id in tar_spk_ids:
    if (
        spk_id
        not in combined_df[combined_df["PARTITION"] == "train"]["TAR_SPK_ID"].values
    ):
        idx = eval_df[eval_df["TAR_SPK_ID"] == spk_id].index[0]
        eval_row = eval_df.loc[idx]
        train_df = pd.concat([train_df, eval_row.to_frame().T])
        eval_df = eval_df.drop(idx)
    if (
        spk_id
        not in combined_df[combined_df["PARTITION"] == "eval"]["TAR_SPK_ID"].values
    ):
        idx = train_df[train_df["TAR_SPK_ID"] == spk_id].index[0]
        train_row = train_df.loc[idx]
        eval_df = pd.concat([eval_df, train_row.to_frame().T])
        train_df = train_df.drop(idx)

# Save the first file with all TAR_SPK_IDs present in both partitions
both_partitions_df = pd.concat([train_df, eval_df])
both_partitions_df.to_csv(overlap_file_path, sep="\t", index=False)

# Filter train and eval to ensure no overlap in TAR_SPK_IDs
train_unique_df = train_df[~train_df["TAR_SPK_ID"].isin(eval_df["TAR_SPK_ID"])]
eval_unique_df = eval_df[~eval_df["TAR_SPK_ID"].isin(train_df["TAR_SPK_ID"])]

# Combine non-overlapping train and eval DataFrames
no_overlap_df = pd.concat([train_unique_df, eval_unique_df])

# Save the second file with non-overlapping TAR_SPK_IDs
no_overlap_df.to_csv(nooverlap_file_path, sep="\t", index=False)

print(
    "Repartitioning complete. Files 'repartitioned_with_overlap.tsv' and 'repartitioned_no_overlap.tsv' have been saved."
)
