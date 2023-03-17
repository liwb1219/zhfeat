#!/bin/bash
python main.py \
--data_name=OneStopE \
--train_dataset_path=data/OneStopE/train.txt \
--valid_dataset_path=data/OneStopE/dev.txt \
--test_dataset_path=data/OneStopE/test.txt \
--seq_len=500 \
--seq_num=2 \
--num_labels=3 \
--rho=0.4 \
--model_dir=/root/BERT