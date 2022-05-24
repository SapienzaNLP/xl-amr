#!/usr/bin/env bash
# Before running this, make sure you already have babelnet mapping of instances in data/cross-lingual-babelnet_mappings/name_span_en_{lang}_map_amr_bn.json and data/misc/verbalization-list-v1.06.txt
set -e

data_dir=$1
util_dir=$2
declare -a StringArray=("en" "ms")
declare -a DatasetArray=("silver" "gold")

printf "files will be written into $util_dir\n"

for lang in ${StringArray[@]}; do
    for dataset_type in ${DatasetArray[@]}; do
        printf "Processing $lang-$dataset_type\n"
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
            
            printf "Create Frame-Lemma Counter...`date`\n"
            python -u -m xlamr_stog.data.dataset_readers.amr_parsing.node_utils \
                --dump_dir ${util_dir} \
                --amr_train_files ${train_data}.input_clean ${dev_data}.input_clean \
                --lang ${lang}

            printf "Done.`date`\n\n"

            printf "Recategorizing subgraphs...`date`\n"
            python -u -m xlamr_stog.data.dataset_readers.amr_parsing.preprocess.recategorizer_multilingual \
                --dump_dir ${util_dir} \
                --amr_train_file ${train_data}.input_clean ${dev_data}.input_clean \
                --amr_files ${train_data}.input_clean ${dev_data}.input_clean \
                --lang ${lang} \
                --build_utils

            printf "Done.`date`\n\n"
        else
            printf "$train_data and $dev_data not found, continuing\n"
        fi
    done
done