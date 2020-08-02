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
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

from model import UniImageRes152
from dataloader import ECImageTestDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/competition/SIGIR2020/Raw_data', help='preprocessed dataset directory')
parser.add_argument('--text_ptm_dir', default='/home/competition/SIGIR2020/pretrained/camembert-base', help='pretrained language model directory')
parser.add_argument('--img_ptm_path', default='/home/competition/SIGIR2020/pretrained/resnet152-b121ed2d.pth', help='pretrained image model path')
parser.add_argument('--result_dir', default='/home/competition/SIGIR2020/Task_1/imgmodal/results', help='directory to store results')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--num_classes', type=int, default=27, help='the number of classes')
parser.add_argument('--img_ft_dim', type=int, default=2048, help='the dim of image features')
parser.add_argument('--text_ft_dim', type=int, default=768, help='the dim of text features')
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print('Loading data...')
    test = pd.read_csv(args.data_dir + '/data/x_test_task1_phase1.tsv', sep = '\t')

    test['Description'] = test['Description'].fillna('')
    sub_columns = ['Title', 'Description', 'Image_id', 'Product_id']
    
    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    test_img_dir = os.path.join(args.data_dir, 'image/image_test/image_test_task1_phase1') 
    test_data = ECImageTestDataset(test[sub_columns].values, test_img_dir, img_transform = img_transform)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, num_workers = 8)

    model = UniImageRes152(args.img_ptm_path, args.img_ft_dim, args.num_classes).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    ckpt = torch.load(args.result_dir + '/best_checkpoint_bs128_lr0.0001_ep26.pth.tar')
    model.load_state_dict(ckpt['state_dict'])
    
    all_preds = inference(test_loader, model)
    Y_test_df = test[['Integer_id', 'Image_id', 'Product_id']] 
    Y_test_df['Prdtypecode'] = all_preds
    
    with open('../data/preprocessed/idx2code.json', 'r') as f:
        idx2code = eval(json.load(f))
    Y_test_df['Prdtypecode'] = Y_test_df['Prdtypecode'].map(idx2code)
    Y_test_df.to_csv(args.result_dir + '/y_test_task1_phase1_pred.tsv', index = False, sep = '\t')
    Y_test_df.to_excel(args.result_dir + '/y_test_task1_phase1_pred.xlsx', index = False, encoding = 'utf-8')
    

def inference(valid_loader, model):
    model.eval()
    all_preds = [] 
    with torch.no_grad():
        for imgs in tqdm(valid_loader):
            imgs = imgs.to(device)
               
            outputs = model(imgs)
            logits = F.softmax(outputs, dim = 1)
            preds = torch.argmax(logits, dim = 1)
            all_preds.extend(preds.data.cpu().numpy())

    return all_preds


if __name__ == '__main__':
    main()
