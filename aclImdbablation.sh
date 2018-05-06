#!/usr/bin/env bash

NAME="aclImdbSimpleAblation"
OUT="temp/$NAME"

# Preprocess the dataset
python preprocess.py --corpus aclImdb_tok --output ${OUT}/data --max_seq_length 3000
python w2v.py --input ${OUT}/data --save_data "demo" --embeddings "vectors_aclImdb.txt"

# 1. Without using pretrained embeddings
python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_wopretrained" \
--nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 20 \
--optim adam --beta1 0.0 --beta2 0.98

# 2. Without finetuning embeddings
python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_wofinetuning" \
--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 20 \
--optim adam --beta1 0.0 --beta2 0.98

# 3. Using larger different words per batch
python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_5kwordsperbatch" \
--use_pretrained_embeddings --train_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 20 \
--optim adam --beta1 0.0 --beta2 0.98 --wbatchsize 5000

# 4. Using smaller different words per batch
python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_1kwordsperbatch" \
--use_pretrained_embeddings --train_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 20 \
--optim adam --beta1 0.0 --beta2 0.98 --wbatchsize 1000

# 5. Using smaller hidden layer
python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_256hidden" \
--use_pretrained_embeddings --train_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 256 --nepochs 20 \
--optim adam --beta1 0.0 --beta2 0.98

# 6. Using 64 size fixed batching strategy
python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_64fixedbatchsize" \
--use_pretrained_embeddings --train_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 20 \
--optim adam --beta1 0.0 --beta2 0.98 --batchsize 64 --batching_strategy "fixed"

# 7. Using 32 size fixed batching strategy
python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_32batchsize" \
--use_pretrained_embeddings --train_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 20 \
--optim adam --beta1 0.0 --beta2 0.98 --batchsize 32 --batching_strategy "fixed"

# 8. Using default adam
python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_defaultAdam" \
--use_pretrained_embeddings --train_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 20 \
--optim adam

# 9. Using different max norm
python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_maxnorm5" \
--use_pretrained_embeddings --train_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 20 \
--optim adam --max_norm 5.0

# 10. Using smaller vocab
python preprocess.py --corpus aclImdb_tok --output ${OUT}/data --max_seq_length 3000 --vocab_size 30000
python w2v.py --input ${OUT}/data --save_data "demo" --embeddings "vectors_aclImdb.txt"

python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_30Kvocab" \
--use_pretrained_embeddings --train_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 20 \
--optim adam --beta1 0.0 --beta2 0.98

# 11. Using Smaller length sequences (This is the same as average of IMDB sentence length)
python preprocess.py --corpus aclImdb_tok --output ${OUT}/data --max_seq_length 300
python w2v.py --input ${OUT}/data --save_data "demo" --embeddings "vectors_aclImdb.txt"

python main.py --corpus aclImdb_tok --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "aclImdb_300maxseqlength" \
--use_pretrained_embeddings --train_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 20 \
--optim adam --beta1 0.0 --beta2 0.98 --scheduler "ExponentialLR"

