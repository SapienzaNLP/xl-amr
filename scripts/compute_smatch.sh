#!/bin/bash

set -e
lang=$1
model_dir=$2
pred=${model_dir}/test_output/test_amr.${lang}.pred.txt
gold=data/AMR/amr_2.0/test_${lang}.txt.features

cp $pred $gold tools/amr-evaluation-tool-enhanced
cd tools/amr-evaluation-tool-enhanced && ./evaluation.sh test_amr.${lang}.pred.txt.frame.wiki.expand  test_${lang}.txt.features

