#!/usr/bin/env bash

NAME="aclImdbSimpleAnalysis_nounlab_entmin"
OUT="temp/$NAME"

# Preprocess the dataset
python preprocess.py --corpus aclImdb_tok --output ${OUT}/data #--max_seq_length 3000

python w2v.py --input ${OUT}/data --save_data "demo" --embeddings "vectors_aclImdb.txt"

python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_nounlab_entmin" \
--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 \
--nepochs 50 --optim adam --beta1 0.0 --beta2 0.98 \
--wbatchsize 3000 --lambda_at 0.0 --lambda_vat 0.0 --lambda_entropy 1.0