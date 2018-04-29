#!/usr/bin/env bash

cd ../..

NAME="aclImdb_25k"
OUT="temp/$NAME"

mkdir -p ${OUT}

# Preprocess
python preprocess.py --corpus aclImdb --output ${OUT}/data --vocab_size 25000 --input /mnt/dataset1/text_classification/data

# Create pre-trained word embeddings
python w2v.py --input ${OUT}/data --save_data "demo" --embeddings "/mnt/word_embeddings/vectors_aclImdb.txt"

# Train the model
PYTHONIOENCODING=utf-8 python main.py --corpus aclImdb --model LSTMEncoder --debug \
--input ${OUT}/data --output_path ${OUT}/model --model_file "ae_training.pt" \
--nepoch_no_imprv 20 --timedistributed --d_hidden 256 --nepochs 50 \
--optim adam --multi_gpu --use_pretrained_embeddings \
--add_noise --noise_dropout 0.1 --lambda_ae 1.0 --lambda_dis 1.0 --lambda_adv 1.0