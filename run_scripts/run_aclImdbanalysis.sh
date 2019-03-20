#!/usr/bin/env bash
NAME="aclImdbSimpleAnalysis"
OUT="temp/$NAME"

# Preprocess the dataset
python preprocess.py --corpus aclImdb_tok --output ${OUT}/data --max_seq_length 3000

python w2v.py --input ${OUT}/data --save_data "demo" --embeddings "vectors_aclImdb.txt"


for d_hidden in 128 256 512 768 1024
do
    python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
    --multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_h${d_hidden}" \
    --use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden ${d_hidden} \
    --nepochs 50 --optim adam --beta1 0.0 --beta2 0.98
done


for num_layers in 1 2 3 4
do
    python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
    --multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_l${num_layers}" \
    --use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --num_layers ${num_layers} \
    --d_hidden 512 --nepochs 50 --optim adam --beta1 0.0 --beta2 0.98
done