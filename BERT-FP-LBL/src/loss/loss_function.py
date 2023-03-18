# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class LengthBalancedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.data_name == 'Cambridge':
            self.register_buffer('lbl', torch.tensor([0.0583 ** config.rho,
                                                      0.113 ** config.rho,
                                                      0.2451 ** config.rho,
                                                      0.3017 ** config.rho,
                                                      0.2818 ** config.rho]).to(config.device))
        elif config.data_name == 'OneStopE':
            self.register_buffer('lbl', torch.tensor([0.2649 ** config.rho,
                                                      0.3315 ** config.rho,
                                                      0.4036 ** config.rho]).to(config.device))
        elif config.data_name == 'WeeBit':
            self.register_buffer('lbl', torch.tensor([0.1051 ** config.rho,
                                                      0.1145 ** config.rho,
                                                      0.1693 ** config.rho,
                                                      0.2565 ** config.rho,
                                                      0.3545 ** config.rho]).to(config.device))

    def forward(self, outputs, target):
        num_classes = outputs.size()[-1]
        target_one_hot = F.one_hot(target, num_classes)
        log_preds = F.log_softmax(outputs, dim=-1)
        loss = -target_one_hot * log_preds
        loss = self.lbl * loss
        loss = loss.sum(dim=-1).mean()
        return loss
