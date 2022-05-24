#!/usr/bin/env bash

set -e

echo "Downloading embeddings."
mkdir -p data/bert-base-multilingual-cased
curl -O https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz
tar -xzvf bert-base-multilingual-cased.tar.gz -C data/bert-base-multilingual-cased
curl -o data/bert-base-multilingual-cased/bert-base-multilingual-cased-vocab.txt \
    https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt
rm bert-base-multilingual-cased.tar.gz

# python -u -m xlamr_stog/data/data_misc/numberbatch_emb.py
# rm data/numberbatch/out_*

echo "Downloading tools."
mkdir -p tools
git clone https://github.com/ChunchuanLv/amr-evaluation-tool-enhanced.git tools/amr-evaluation-tool-enhanced

# curl -O http://www.airpedia.org/tint/0.2/tint-runner-0.2-bin.tar.gz
# tar -xzvf tint-runner-0.2-bin.tar.gz -C tools
# rm tint-runner-0.2-bin.tar.gz


