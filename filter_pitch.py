import pandas as pd

# Load the first file (PITCH values)
pitch_df = pd.read_csv('data/Database/utt2pitch', sep=' ', header=None, names=['ASVSPOOF_ID', 'PITCH'])

# Load the second file (metadata)
metadata_df = pd.read_csv('data/Database/ASVspoof_VCTK_aligned_physical_meta_partitioned_eval_bonafide_no_overlap.tsv', sep='\t', header=None, names=['ASVSPOOF_ID', "DURATION", "PITCH", "SNR", "SPK_RATE", "PARTITION"])

print(metadata_df.columns)

# Replace the "PITCH" column in the metadata with values from pitch_df
metadata_df = metadata_df.merge(pitch_df, on='ASVSPOOF_ID', how='left', suffixes=('', '_new'))

# Update the PITCH column in the metadata file
metadata_df['PITCH'] = metadata_df['PITCH_new'].fillna(metadata_df['PITCH'])

# Drop the temporary column used for merging
metadata_df = metadata_df.drop(columns=['PITCH_new'])

# Save the updated metadata back to a file
metadata_df.to_csv('data/Database/ASVspoof_VCTK_aligned_physical_meta_partitioned_eval_bonafide_no_overlap_pitch_fixed.tsv', sep='\t', index=False, header=True)

