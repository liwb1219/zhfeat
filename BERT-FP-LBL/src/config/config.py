# coding=utf-8
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()  # 创建一个解析对象
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # ============================== Data Config ==============================
    parser.add_argument('--data_name', type=str, default='OneStopE')
    parser.add_argument('--train_dataset_path', type=str, default='data/OneStopE/train.txt')
    parser.add_argument('--valid_dataset_path', type=str, default='data/OneStopE/dev.txt')
    parser.add_argument('--test_dataset_path', type=str, default='data/OneStopE/test.txt')

    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--seq_num', type=int, default=2)

    # ============================== Model Config ==============================
    parser.add_argument('--model_dir', type=str, default='/root/workspace/BERT-family/BERT')
    parser.add_argument('--mix_lambda', type=float, default=0.5)
    parser.add_argument('--linguistic_features_size', type=int, default=327)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--intermediate_size', type=str, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_labels', type=int, default=3)

    # ============================== Loss Config ==============================
    parser.add_argument('--rho', type=float, default=0.0)

    # ============================== Train Config ==============================
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # ============================== Logger Config ==============================
    parser.add_argument('--log_file', type=str, default='worker.log')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--log_file_mode', type=str, default='a')

    # ============================== Checkpoint Config ==============================
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--ckpt_file', type=str, default='save/__model__.bin')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for arg in vars(args):
        print(arg + ':', getattr(args, arg))
