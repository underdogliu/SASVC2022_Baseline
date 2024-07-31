#!/bin/sh
# Generate the pitch energy

set -e
. ./kaldi_path.sh

datadir=$1
nj=20

mfccdir=$datadir/mfccs

utils/fix_data_dir.sh ${datadir} || exit 1;

steps/make_mfcc_pitch.sh --mfcc-config scripts/asv_cm_explainability/physics_labels/mfcc_hires.conf \
    --pitch-config scripts/asv_cm_explainability/physics_labels/pitch.conf \
    --nj $nj $datadir exp/make_mfcc/ ${mfccdir}

utils/data/limit_feature_dim.sh 40:43 $datadir ${datadir}/pitch
utils/fix_data_dir.sh ${datadir}/pitch || exit 1;

python3 scripts/asv_cm_explainability/physics_labels/write_utt2pitch.py \
    ${datadir}/pitch/feats.scp ${datadir}
