#!/usr/bin/env bash

set -e

gold_path=data/AMR/panl_en_ms
dir=data/AMR/panl_en_ms_out
python -u -m xlamr_stog.data.dataset_readers.amr_projection.project_par_train_dev --amr_path ${gold_path} --out_path ${dir}/ --lang id

echo "Done!"