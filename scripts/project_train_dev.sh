#!/usr/bin/env bash

set -e

dir=data/AMR/amr_2.0
out_path=data/AMR/amr_2.0_zh_de_es_it
translations=${out_path}/translations

python -u -m xlamr_stog.data.dataset_readers.amr_projection.project_train_dev \
        --amr_path ${dir} --trans_path ${translations} --out_path ${out_path}/ \


