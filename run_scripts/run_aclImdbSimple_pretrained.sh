#!/usr/bin/env bash

NAME="aclImdbSimple_pretrained"
OUT="temp/$NAME"

mkdir -p ${OUT}

# Preprocess
# python preprocess.py --corpus aclImdb_tok --output ${OUT}/data


# Create pre-trained word embeddings
# python w2v.py --input ${OUT}/data --save_data "demo" --embeddings vectors_aclImdb.txt # "${HOME}/Desktop/word_embeddings/vectors_aclImdb.txt"


# Train the model
python -m pdb main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdbSimple_pretrained" \
--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 50 \
--optim adam --beta1 0.0 --beta2 0.98