#!/usr/bin/env bash

corpus=$1
dataset_suffix=$2
lambda_entropy=$3
lambda_at=$4
lambda_vat=$5

NAME="${corpus}.${dataset_suffix}_ent.${lambda_entropy}_at.${lambda_at}_vat.${lambda_vat}"
OUT="temp/$NAME"

mkdir -p ${OUT}

# Preprocess
python preprocess.py --corpus elec --elec_train "${corpus}/train.txt.${dataset_suffix}" --output ${OUT}/data --vocab_size 40000 --save_data "demo_fastText"

# Create pre-trained word embeddings
python w2v.py --input ${OUT}/data --save_data "demo_fastText" --embeddings "${HOME}/crawl-300d-2M.vec"

# Train the model
PYTHONIOENCODING=utf-8 python main.py --corpus elec --model LSTMEncoder --debug --save_data "demo_fastText" \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name $NAME \
--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 50 \
--optim adam --wbatchsize 2000 --wbatchsize_unlabel 2000 --eval_steps 700 --lstm_dropout 0.5 --word_dropout 0.5 \
--beta1 0.0 --beta2 0.98 \
--perturb_norm_length 2 \
--lambda_entropy ${lambda_entropy} \
--lambda_vat ${lambda_vat} \
--lambda_at ${lambda_at}
