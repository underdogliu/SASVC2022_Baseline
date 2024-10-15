import pandas as pd

# Load the first TSV file into a DataFrame
tsv_file = "data/Database/ASVspoof_VCTK_aligned_physical_meta_partitioned.tsv"
df = pd.read_csv(tsv_file, sep='\t')

# Load the second file into a DataFrame
second_file = "data/Database/asvspoof2019_trials_eval_bonafide.txt"
df2 = pd.read_csv(second_file, sep=' ', header=None, usecols=[1])
df2.columns = ['ASVSPOOF_ID']

# Filter the first DataFrame based on IDs in the second DataFrame
filtered_df = df[df['ASVSPOOF_ID'].isin(df2['ASVSPOOF_ID'])]

# Save the filtered DataFrame to a new TSV file
filtered_df.to_csv("data/Database/ASVspoof_VCTK_aligned_physical_meta_partitioned_eval_bonafide.tsv", sep='\t', index=False)

print("Filtering complete. The filtered data has been saved to 'ASVspoof_VCTK_aligned_meta_partitioned_eval_bonafide.tsv'.")

