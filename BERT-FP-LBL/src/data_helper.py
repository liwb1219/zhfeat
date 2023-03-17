# coding=utf-8
import ast
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class DatasetReader(Dataset):
    def __init__(self, config, dataset_path):
        label_list = []
        linguistic_features_list = []
        text_list = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for data in f:
                data_dict = ast.literal_eval(data)
                label_list.append(int(data_dict['label']) - 1)
                linguistic_features_list.append(data_dict['linguistic_features'])
                text = re.sub(r'\n', ' ', data_dict['text'])
                word_list = text.split()
                word_list = word_list[0:config.seq_len * config.seq_num]
                for idx in range(config.seq_num):
                    text_list.append(' '.join(word_list[idx * config.seq_len:(idx + 1) * config.seq_len]))

        tokenizer = BertTokenizer.from_pretrained(config.model_dir)
        input_ids = tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=config.seq_len + 2)

        self.input_ids = torch.tensor(input_ids['input_ids']).view(-1, config.seq_num, config.seq_len + 2)
        self.attention_mask = torch.tensor(input_ids['attention_mask']).view(-1, config.seq_num, config.seq_len + 2)
        self.token_type_ids = torch.tensor(input_ids['token_type_ids']).view(-1, config.seq_num, config.seq_len + 2)
        self.labels = torch.tensor(label_list)
        self.linguistic_features = torch.tensor(linguistic_features_list)

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item], self.token_type_ids[item], \
               self.labels[item], self.linguistic_features[item]

    def __len__(self):
        return len(self.input_ids)


def create_dataloader(config):
    train_dataset = DatasetReader(config=config, dataset_path=config.train_dataset_path)
    valid_dataset = DatasetReader(config=config, dataset_path=config.valid_dataset_path)
    test_dataset = DatasetReader(config=config, dataset_path=config.test_dataset_path)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size)
    return train_loader, valid_loader, test_loader
