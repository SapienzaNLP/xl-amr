#!/usr/bin/env bash

set -e

data_dir=$1
dataset_type=$2
declare -a StringArray=("en" "ms")
util_dir=data/AMR/en_ms_utils

for lang in ${StringArray[@]}; do
  echo $lang
  # Data with **features** (.txt.feature)
  if [ "$lang" = "en" ] && [ "$dataset_type" = "gold" ]; then
    train_data=${data_dir}/train.txt.features
    dev_data=${data_dir}/dev.txt.features
  else
    train_data=${data_dir}/train_${lang}.txt.features
    dev_data=${data_dir}/dev_${lang}.txt.features

  fi

  if [[ -f "$train_data" ]] && [[ -f "$dev_data" ]]; then

    printf "Cleaning inputs...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner \
        --amr_files ${train_data} ${dev_data}
    printf "Done.`date`\n\n"

    printf "Recategorizing subgraphs...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.recategorizer_multilingual \
        --dump_dir ${util_dir} \
        --amr_files ${train_data}.input_clean ${dev_data}.input_clean \
        --lang ${lang}

    printf "Done.`date`\n\n"

    printf "Removing senses...`date`\n"
    python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.sense_remover \
        --util_dir ${util_dir} \
        --amr_files ${train_data}.input_clean.recategorize \
        ${dev_data}.input_clean.recategorize
    printf "Done.`date`\n\n"

    printf "Renaming preprocessed files...`date`\n"
    mv ${train_data}.input_clean.recategorize.nosense ${train_data}.recat
    mv ${dev_data}.input_clean.recategorize.nosense ${dev_data}.recat
    rm ${data_dir}/*.input_clean*
  else
    printf "$train_data and $dev_data not found, continuing"
  fi

done