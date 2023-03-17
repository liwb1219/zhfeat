#!/bin/bash
python main.py \
--data_name=WeeBit \
--train_dataset_path=data/WeeBit/train.txt \
--valid_dataset_path=data/WeeBit/dev.txt \
--test_dataset_path=data/WeeBit/test.txt \
--seq_len=510 \
--seq_num=1 \
--num_labels=5 \
--rho=0.8 \
--model_dir=/root/BERT