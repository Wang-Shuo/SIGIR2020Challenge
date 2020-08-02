#!/usr/bin/env python37
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torchvision.models as models
from transformers import CamembertModel

class TextOnly(nn.Module):
    def __init__(self, text_ptm_dir, text_ft_dim, num_classes):
        super(TextOnly, self).__init__()
        self.camembert = CamembertModel.from_pretrained(text_ptm_dir)
        self.classifier = nn.Linear(text_ft_dim, num_classes, bias = True)

    def forward(self, input_ids, attention_mask = None, token_type_ids = None):
        outputs = self.camembert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        # seq_output = outputs[0]
        # pooled_output = torch.mean(seq_output, dim = 1)
        pooled_output = outputs[1]
        output = self.classifier(pooled_output)
        return output   
    
