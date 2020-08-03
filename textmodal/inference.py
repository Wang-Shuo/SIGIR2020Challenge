#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on May 22, 2020
@author: wangshuo
@desc: inference function for SIGIR 2020 challenge.

"""

import os
import json
import time
import argparse
import pickle
import numpy as np
import random
from tqdm import tqdm
from os.path import join
import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

from model import TextOnly
from dataloader import ECTextTestDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/', help='dataset directory')
parser.add_argument('--text_ptm_dir', default='../pretrained/camembert-base', help='pretrained language model directory')
parser.add_argument('--result_dir', default='results/', help='directory to store results')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--num_classes', type=int, default=27, help='the number of classes')
parser.add_argument('--text_ft_dim', type=int, default=768, help='the dim of text features')
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print('Loading data...')
    test = pd.read_csv(args.data_dir + 'preprocessed/x_test_task1_phase1_clean.tsv', sep = '\t')

    test['Description'] = test['Description'].fillna('')
    sub_columns = ['Title', 'Description', 'Image_id', 'Product_id']
    
    test_data = ECTextTestDataset(test[sub_columns].values, args.text_ptm_dir)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, num_workers = 8)

    model = TextOnly(args.text_ptm_dir, args.text_ft_dim, args.num_classes).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1])
    ckpt = torch.load(args.result_dir + '/best_checkpoint_bs128_lr5e-05_ep30_numiter17910_warmup1791.pth.tar')
    model.load_state_dict(ckpt['state_dict'])
    
    all_preds = inference(test_loader, model)
    Y_test_df = test[['Integer_id', 'Image_id', 'Product_id']] 
    Y_test_df['Prdtypecode'] = all_preds
    
    with open('../data/preprocessed/idx2code.json', 'r') as f:
        idx2code = eval(json.load(f))
    Y_test_df['Prdtypecode'] = Y_test_df['Prdtypecode'].map(idx2code)
    Y_test_df.to_csv(args.result_dir + '/y_test_task1_phase1_pred.tsv', index = False, sep = '\t')
     

def inference(valid_loader, model):
    model.eval()
    all_preds = [] 
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids in tqdm(valid_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
           
            outputs = model(input_ids, attention_mask, token_type_ids)
            logits = F.softmax(outputs, dim = 1)
            preds = torch.argmax(logits, dim = 1)
            all_preds.extend(preds.data.cpu().numpy())

    return all_preds


if __name__ == '__main__':
    main()
