#!/usr/bin/env bash

set -e

dir=$1
lang=$2
dataset_type=$3

# For English start a Stanford CoreNLP server before running this script.
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

# For Italian start a Tint server before running this script.
# http://tint.fbk.eu/download.html

# The compound file is downloaded from
# https://github.com/ChunchuanLv/AMR_AS_GRAPH_PREDICTION/blob/master/data/joints.txt
compound_file=data/misc/joints.txt
if [ "$dataset_type" = "gold" ]; then
  if [ "$lang" = "en" ]; then
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator \
        ${dir}/test.txt ${dir}/train.txt ${dir}/dev.txt \
        --compound_file ${compound_file}
  else
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator_multilingual \
        ${dir}/test_${lang}.txt --lang ${lang}
  fi
else
  python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator_multilingual \
      ${dir}/train_${lang}.txt ${dir}/dev_${lang}.txt \
      --lang ${lang}
fi
