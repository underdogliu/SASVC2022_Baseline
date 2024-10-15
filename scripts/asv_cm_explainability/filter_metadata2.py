import pandas as pd

# Load the first and second files into DataFrames
first_file_path = 'data/Database/ASVspoof_VCTK_aligned_meta_partitioned_eval_bonafide_overlap.tsv'
second_file_path = 'data/Database/ASVspoof_VCTK_aligned_physical_meta_partitioned_eval_bonafide.tsv'

df1 = pd.read_csv(first_file_path, sep='\t')
df2 = pd.read_csv(second_file_path, sep='\t', header=None, names=['ASVSPOOF_ID', 'Col2', 'Col3', 'Col4', 'Col5', 'PARTITION'])

# Filter df2 based on the ASVSPOOF_IDs and PARTITION in df1
filtered_df = pd.merge(df2, df1[['ASVSPOOF_ID', 'PARTITION']], on=['ASVSPOOF_ID', 'PARTITION'])

# Save the filtered data to a new file
filtered_df.to_csv('data/Database/ASVspoof_VCTK_aligned_physical_meta_partitioned_eval_bonafide_overlap.tsv', sep='\t', index=False, header=True)

print("Filtered file has been saved as 'filtered_second_file.tsv'.")

