#!/bin/bash
python main.py \
--data_name=Cambridge \
--train_dataset_path=data/Cambridge/train.txt \
--valid_dataset_path=data/Cambridge/dev.txt \
--test_dataset_path=data/Cambridge/test.txt \
--seq_len=500 \
--seq_num=2 \
--num_labels=5 \
--rho=0.6 \
--model_dir=/root/BERT