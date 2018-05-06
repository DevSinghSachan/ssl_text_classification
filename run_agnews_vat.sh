#!/usr/bin/env bash
NAME="agnews_pretrained_wiki"
OUT="temp/$NAME"

mkdir -p ${OUT}

# Preprocess
# python preprocess.py --corpus agnews --output ${OUT}/data --vocab_size 75000 --save_data "demo_fastText"

# Create pre-trained word
# python w2v.py --input ${OUT}/data --save_data "demo_fastText" --embeddings "${HOME}/crawl-300d-2M.vec"

# Train the model
PYTHONIOENCODING=utf-8 python main.py --corpus agnews --model LSTMEncoder --debug --save_data "demo_fastText" \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "elec_clf_vat" \
--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 30 \
--optim adam --wbatchsize 2000 --wbatchsize_unlabel 2000 --eval_steps 1000 --lstm_dropout 0.5 --word_dropout 0.5 --beta1 0.0 --num_layers 1 --beta2 0.98 --scheduler "ExponentialLR" --lambda_vat 1.0 --lambda_entropy 0.0 --lambda_at 1.0  # --wbatchsize 1500 --wbatchsize_unlabel 1500