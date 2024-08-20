import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.special import kl_div


def load_embeddings(embedding_folder, asvspoof_id):
    """Load the numpy embedding file corresponding to the ASVSPOOF_ID."""
    file_path = os.path.join(embedding_folder, f"{asvspoof_id}.npy")
    return np.load(file_path)


def itakura_saito_distance(p, q):
    """Compute Itakura-Saito distance between two 2-D spectrograms."""
    return np.sum(kl_div(p, q) - np.log(kl_div(p, q) + 1))


def compute_distances(embeddings1, embeddings2):
    """Compute the distance between two sets of embeddings."""
    if embeddings1.ndim == 1 and embeddings2.ndim == 1:
        return cosine(embeddings1, embeddings2)
    elif embeddings1.ndim == 2 and embeddings2.ndim == 2:
        return itakura_saito_distance(embeddings1, embeddings2)
    else:
        raise ValueError("Embedding dimensions must match for comparison.")


def process_gender(embedding_folder, trial_file, gender_metadata, gender_label):
    """Process embeddings for a specific gender, compute distances and save results."""
    distance_file = f"{gender_label}_distance.txt"
    with open(distance_file, "w") as f_out:
        for spk_id in gender_metadata["TAR_SPK_ID"].unique():
            speaker_trials = trial_file[trial_file["spk"] == spk_id]
            bonafide_ids = speaker_trials[speaker_trials["decision"] == "bonafide"][
                "ASVSPOOF_ID"
            ]
            spoof_ids = speaker_trials[speaker_trials["decision"] == "spoof"][
                "ASVSPOOF_ID"
            ]

            if bonafide_ids.empty or spoof_ids.empty:
                continue

            # Load embeddings
            bonafide_embeddings = [
                load_embeddings(embedding_folder, asv_id) for asv_id in bonafide_ids
            ]
            spoof_embeddings = [
                load_embeddings(embedding_folder, asv_id) for asv_id in spoof_ids
            ]

            # Compute pairwise distances between bonafide and spoof embeddings
            distances = [
                compute_distances(b_emb, s_emb)
                for b_emb in bonafide_embeddings
                for s_emb in spoof_embeddings
            ]

            # Average distance for the speaker
            avg_distance = np.mean(distances)
            f_out.write(f"{spk_id} {avg_distance}\n")

    return distance_file


def plot_distances(male_distance_file, female_distance_file):
    """Plot the distance distribution for both male and female embeddings."""
    male_distances = np.loadtxt(male_distance_file, usecols=[1])
    female_distances = np.loadtxt(female_distance_file, usecols=[1])

    plt.figure(figsize=(8, 6))
    plt.hist(
        male_distances,
        bins=30,
        alpha=0.5,
        label="Male",
        color="blue",
        linewidth=2,
        density=True,
    )
    plt.hist(
        female_distances,
        bins=30,
        alpha=0.5,
        label="Female",
        color="red",
        linewidth=2,
        density=True,
    )

    plt.xlabel("Distance", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Distance Distribution by Gender", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gender_distance_distribution.png", dpi=300)
    plt.show()


def main(embedding_folder, metadata_file, trial_file_path):
    # Load metadata and trial files
    metadata = pd.read_csv(metadata_file, sep="\t")
    trial_file = pd.read_csv(
        trial_file_path,
        sep=" ",
        names=["spk", "ASVSPOOF_ID", "col3", "col4", "decision"],
    )

    # Split metadata by gender
    male_metadata = metadata[metadata["GENDER"] == "M"]
    female_metadata = metadata[metadata["GENDER"] == "F"]

    # Process and compute distances for male and female speakers
    male_distance_file = process_gender(
        embedding_folder, trial_file, male_metadata, "male"
    )
    female_distance_file = process_gender(
        embedding_folder, trial_file, female_metadata, "female"
    )

    # Plot the distributions
    plot_distances(male_distance_file, female_distance_file)


if __name__ == "__main__":
    embedding_folder = sys.argv[1]
    metadata_file = sys.argv[2]
    trial_file_path = sys.argv[3]

    main(embedding_folder, metadata_file, trial_file_path)
