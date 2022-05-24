#!/usr/bin/env bash

set -e

gold_path=data/AMR/amr_2.0
dir=data/AMR/amr_2.0_ms
translations=${dir}/translations
python -u -m xlamr_stog.data.dataset_readers.amr_projection.project_train_dev_unfiltered --amr_path ${gold_path} --trans_path ${translations} --out_path ${dir}/ \

echo "Done!"