#!/usr/bin/env bash

set -e

dir=data/AMR/amr_2.0
translations=${dir}/translations
gold_en=${dir}/test.txt
python -u -m xlamr_stog.data.dataset_readers.amr_projection.project_test --amr_test ${gold_en} --trans_path ${translations} --out_path ${dir}/ \

echo "Done!"
