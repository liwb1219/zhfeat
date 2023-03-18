# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BERTFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.bert = BertModel.from_pretrained(config.model_dir)
        self.mix_lambda = nn.Parameter(torch.tensor(config.mix_lambda))
        self.max_pooling = nn.MaxPool2d(kernel_size=(config.seq_num, 1), stride=1)
        self.avg_pooling = nn.AvgPool2d(kernel_size=(config.seq_num, 1), stride=1)
        self.nnf_dim_transform = nn.Linear(config.hidden_size, config.intermediate_size)
        self.af_dim_transform = nn.Linear(config.linguistic_features_size, config.intermediate_size)
        self.dual_feature_transform = nn.Linear(config.intermediate_size, config.intermediate_size)

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.intermediate_size * 2, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, linguistic_features):
        # input_ids = [batch_size, seq_num, seq_len]
        batch_size, seq_num, seq_len = input_ids.size()
        input_ids = input_ids.view(batch_size * seq_num, seq_len)
        token_type_ids = token_type_ids.view(batch_size * seq_num, seq_len)
        attention_mask = attention_mask.view(batch_size * seq_num, seq_len)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = pooled_output.view(batch_size, seq_num, -1)

        max_pooling_output = self.max_pooling(pooled_output).squeeze(1)
        avg_pooling_output = self.avg_pooling(pooled_output).squeeze(1)

        mix_pooling_output = self.mix_lambda * max_pooling_output + (1 - self.mix_lambda) * avg_pooling_output

        neural_network_features = F.tanh(self.nnf_dim_transform(mix_pooling_output))
        linguistic_features = F.tanh(self.af_dim_transform(linguistic_features))

        neural_network_features = F.tanh(self.dual_feature_transform(neural_network_features))
        linguistic_features = F.tanh(self.dual_feature_transform(linguistic_features))

        batch_size, _ = neural_network_features.size()
        orthogonal_features = torch.zeros(batch_size, _).to(self.device)
        for i in range(batch_size):
            orthogonal_features[i] = self.get_orthogonal_vector(neural_network_features[i], linguistic_features[i])

        fusion_features = torch.cat((neural_network_features, orthogonal_features), dim=-1)

        fusion_features = self.dropout(fusion_features)
        logits = self.classifier(fusion_features)
        return logits

    @staticmethod
    def get_orthogonal_vector(main_vector, sub_vector):
        projection_vector = main_vector * torch.dot(main_vector, sub_vector) / torch.dot(main_vector, main_vector)
        orthogonal_vector = sub_vector - projection_vector
        return orthogonal_vector
