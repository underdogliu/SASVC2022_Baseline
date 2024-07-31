"""
Average the pitch features and store them into the numpy tarball
"""

import kaldi_io
import numpy as np
import os
import sys


if __name__ == "__main__":
    scp_file = sys.argv[1]
    tar_dir = sys.argv[2]
    os.makedirs(tar_dir, exist_ok=True)
    with open(tar_dir + "/utt2pitch", "w") as t:
        for key, mat in kaldi_io.read_mat_scp(scp_file):
            vec = np.mean(mat, axis=0)
            pitch_energy = vec[0]
            t.write("{} {}\n".format(key, pitch_energy))