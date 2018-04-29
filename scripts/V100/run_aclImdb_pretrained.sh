#!/usr/bin/env bash

cd ../..

NAME="aclImdbSimple"
OUT="temp/$NAME"

mkdir -p ${OUT}

# Preprocess
python preprocess.py --corpus aclImdb --output ${OUT}/data --input /mnt/dataset1/text_classification/data

# Create pre-trained word embeddings
python w2v.py --input ${OUT}/data --save_data "demo" --embeddings "/mnt/word_embeddings/vectors_aclImdb.txt"

# Train the model
PYTHONIOENCODING=utf-8 python main.py --corpus aclImdb --model LSTMEncoder --debug \
--input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_pretrained" \
--nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 50 \
--optim adam --beta1 0.0 --beta2 0.98 --multi_gpu --use_pretrained_embeddings \
--adaptive_dropout --lstm_dropout 0.0 --word_dropout 0.2
# --add_noise --noise_dropout 0.3 --word_dropout 0.0 --lstm_dropout 0.0
# --adaptive_dropout --lstm_dropout 0.0 --normalize_embedding