#!/usr/bin/env bash

set -e

# Directory where intermediate utils will be saved to speed up processing.
util_dir=data/AMR/en_ms_utils

lang=$1
model_dir=$2
test_data=${model_dir}/test_output/test_amr.${lang}.pred.txt

spotlight_path=${util_dir}/spotlight/${lang}_test_spotlight_wiki.json

# ========== Set the above variables correctly ==========

printf "Frame lookup...`date`\n"
python -u -m xlamr_stog.data.dataset_readers.amr_parsing.postprocess.node_restore \
    --amr_files ${test_data} \
    --util_dir ${util_dir}
printf "Done.`date`\n\n"

printf "Wikification...`date`\n"
python -u -m xlamr_stog.data.dataset_readers.amr_parsing.postprocess.wikification \
    --amr_files ${test_data}.frame \
    --util_dir ${util_dir}\
    --spotlight_wiki $spotlight_path\
    --lang ${lang}\
    --exclude_spotlight
printf "Done.`date`\n\n"

printf "Expanding nodes...`date`\n"
python -u -m xlamr_stog.data.dataset_readers.amr_parsing.postprocess.expander \
    --amr_files ${test_data}.frame.wiki \
    --util_dir ${util_dir} \
    --u_pos True \
    --lang ${lang}

printf "Done.`date`\n\n"

mv ${test_data}.frame.wiki.expand ${test_data}.postproc
rm ${test_data}.frame*
