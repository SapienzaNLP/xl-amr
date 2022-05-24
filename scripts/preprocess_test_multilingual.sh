#!/usr/bin/env bash

set -e

data_dir=data/AMR/amr_2.0
declare -a StringArray=("en" "ms" )
util_dir=data/AMR/en_ms_utils

for lang in ${StringArray[@]}; do
  echo $lang
  # Data with **features** (.txt.feature)
  if [ "$lang" = "en" ]; then
    test_data=${data_dir}/test.txt.features
  else
    test_data=${data_dir}/test_${lang}.txt.features

  fi

  if [[ -f "$test_data" ]]; then

    # ========== Set the above variables correctly ==========

    printf "Cleaning inputs...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner \
        --amr_files ${test_data}
    printf "Done.`date`\n\n"

    printf "Anonymizing NM ...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.text_anonymizor \
        --util_dir ${util_dir} \
        --amr_file ${test_data}.input_clean \
        --lang ${lang}

    printf "Done.`date`\n\n"

    printf "Removing senses...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.sense_remover \
        --util_dir ${util_dir} \
        --amr_files ${test_data}.input_clean.recategorize \

    printf "Done.`date`\n\n"

    printf "Renaming preprocessed files...`date`\n"
    mv ${test_data}.input_clean.recategorize.nosense ${test_data}.recat
    rm ${data_dir}/*.input_clean*

  fi

done