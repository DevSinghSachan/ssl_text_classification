#!/usr/bin/env bash

NAME="elec_vat"
OUT="temp/$NAME"

mkdir -p ${OUT}

# Preprocess
python preprocess.py --corpus elec --output ${OUT}/data --vocab_size 40000 --save_data "demo_fastText" # --vocab_size 150000 --max_seq_length 3000


# Create pre-trained word embeddings
python w2v.py --input ${OUT}/data --save_data "demo_fastText" --embeddings "${HOME}/crawl-300d-2M.vec" # "vectors_elec.txt"

exit
# Train the model
PYTHONIOENCODING=utf-8 python main.py --corpus elec --model LSTMEncoder --debug --save_data "demo_fastText" \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "elec_clf_entropy_vat" \
--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 50 \
--optim adam --wbatchsize 2000 --wbatchsize_unlabel 2000 --eval_steps 700 --lstm_dropout 0.5 --word_dropout 0.5 --beta1 0.0 --beta2 0.98 --lambda_entropy 1.0 --lambda_vat 1.0 --lambda_at 1.0 # --wbatchsize 1500 --wbatchsize_unlabel 1500