#!/usr/bin/env bash

set -e

amr_dict=$1
out_dict=$2

printf "Extracting sentences from AMR in ${amr_dict} into ${out_dict}\n"
mkdir -p ${out_dict}

grep "# ::snt" ${amr_dict}/dev.txt | sed "s/# ::snt //g" > ${out_dict}/dev.snt.txt
grep "# ::snt" ${amr_dict}/train.txt | sed "s/# ::snt //g" > ${out_dict}/train.snt.txt