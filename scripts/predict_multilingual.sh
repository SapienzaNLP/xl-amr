#!/usr/bin/env bash
cuda=2
lang=$1
model_dir=$2

if [[ "$lang" = "en" ]]; then
  input_file=data/AMR/amr_2.0/test.txt.features.recat
else
  input_file=data/AMR/amr_2.0/test_${lang}.txt.features.recat
fi

printf "Predicting...`date`\n"
python -u -m xlamr_stog.commands.predict --archive-file ${model_dir}/ \
--weights-file ${model_dir}/best.th \
--input-file $lang $input_file \
--batch-size 32 \
--use-dataset-reader \
--cuda-device $cuda \
--output-file ${model_dir}/test_output/test_amr.${lang}.pred.txt \
--silent \
--beam-size 5 \
--predictor STOG

printf "Done.`date`\n\n"