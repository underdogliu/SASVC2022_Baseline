#!/bin/sh

datadir=$1
nj=20


utils/fix_data_dir.sh ${datadir} || exit 1;

utils/data/get_utt2dur.sh ${datadir}} 1>&2 || exit 1;
