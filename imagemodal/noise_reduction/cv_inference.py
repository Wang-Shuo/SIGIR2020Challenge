#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on July 15, 2020
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

from model import UniImageResNet50
from dataloader import ECImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='/home/competition/SIGIR2020/', help='preprocessed dataset directory')
parser.add_argument('--img_ptm_path', default='/home/competition/SIGIR2020/pretrained/resnet50-19c8e357.pth', help='pretrained image model path')
parser.add_argument('--result_dir', default='/home/competition/SIGIR2020/Task_1/imgmodal/cv4noise/results', help='directory to store results')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--num_classes', type=int, default=27, help='the number of classes')
parser.add_argument('--img_ft_dim', type=int, default=2048, help='the dim of image features')
args = parser.parse_args()
print(args)

#device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def main():
    print('Loading data...')
    for i in range(4):
        test = pd.read_csv(args.root_dir + 'Task_1/data/preprocessed/cv/valid_cv_{}.tsv'.format(i), sep = '\t')

        test['Description'] = test['Description'].fillna('')
        sub_columns = ['Title', 'Description', 'Image_id', 'Product_id', 'Prdtypecode']

        img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        test_img_dir = os.path.join(args.root_dir, 'Raw_data/image/image_training') 
        test_data = ECImageDataset(test[sub_columns].values, test_img_dir, img_transform = img_transform)
        test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, num_workers = 8)

        model = UniImageResNet50(args.img_ptm_path, args.num_classes).to(device)
        ckpt = torch.load(args.result_dir + '/best_checkpoint_cv{}_seed1024_bs128_lr0.01_ep50.pth.tar'.format(i), map_location = 'cpu')
        model.load_state_dict(ckpt['state_dict'])

        result_df = inference(test_loader, model)
        result_df = pd.concat([test[['Integer_id', 'Image_id', 'Product_id', 'Prdtypecode']], result_df], axis = 1)
        result_df.to_csv(args.result_dir + '/cv{}_prob_result.tsv'.format(i), index = False, sep = '\t')

    

def inference(valid_loader, model):
    model.eval()
    all_preds = [] 
    with torch.no_grad():
        for imgs, labels in tqdm(valid_loader):
            imgs = imgs.to(device)
               
            outputs = model(imgs)
            logits = F.softmax(outputs, dim = 1)
            preds = torch.argmax(logits, dim = 1)
            results = torch.cat([logits, preds.unsqueeze(1).float()], dim = 1)
            all_preds.extend(results.data.cpu().numpy())

    result_df = pd.DataFrame(all_preds, columns = ['class_{}_prob'.format(i) for i in range(args.num_classes)] + ['class_pred'])

    return result_df


if __name__ == '__main__':
    main()
