#!/usr/bin/env bash

NAME="quora_pretrained"
OUT="temp/$NAME"

mkdir -p ${OUT}

# Preprocess
python preprocess.py --corpus quora_iq --output ${OUT}/data


# Create pre-trained word embeddings
python w2v.py --input ${OUT}/data --save_data "demo" --embeddings "${HOME}/crawl-300d-2M.vec"


# Train the model
python main.py --corpus quora_iq --model LSTMEncoder --debug \
--input ${OUT}/data --output_path ${OUT}/model --exp_name "quora_pretrained" \
--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 256 --nepochs 50 \
--optim adam --beta1 0.0 --beta2 0.98
