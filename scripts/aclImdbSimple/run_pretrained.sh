#!/usr/bin/env bash

cd ../..

NAME="aclImdbSimple_tok"
OUT="temp/$NAME"

mkdir -p ${OUT}

# Preprocess
# python preprocess.py --corpus aclImdb_tok --output ${OUT}/data

# Create pre-trained word embeddings
# python w2v.py --input ${OUT}/data --save_data "demo" --embeddings "${HOME}/Desktop/word_embeddings/vectors_aclImdb.txt"

# Train the model
PYTHONIOENCODING=utf-8 python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --model_file "pretrained_training.pt" \
--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 50 \
--optim adam --beta1 0.0 --beta2 0.98
# --add_noise --noise_dropout 0.3 --word_dropout 0.0 --lstm_dropout 0.0
# --adaptive_dropout --lstm_dropout 0.0 --normalize_embedding