#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on May 30, 2020
@author: wangshuo
@desc: train script with only text modal using multi-gpu for SIGIR 2020 challenge Task 1.

"""

import os
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
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


from model import TextOnly
from dataloader import ECTextDataset
from utils import FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/competition/SIGIR2020/Task_1/data/preprocessed/', help='preprocessed dataset directory')
parser.add_argument('--text_ptm_dir', default='/home/biye/Pretrained_models/camembert-large', help='pretrained language model directory')
# parser.add_argument('--text_ptm_dir', default='/home/competition/SIGIR2020/pretrained/camembert-base', help='pretrained language model directory')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--num_classes', type=int, default=27, help='the number of classes')
parser.add_argument('--text_ft_dim', type=int, default=1024, help='the dim of text features')
parser.add_argument('--result_dir', default='/home/competition/SIGIR2020/Task_1/textmodal/results', help='directory to store results')
parser.add_argument('--epoch', type=int, default=40, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=10, help='the number of steps after which the learning rate decay')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--seed', type=int, default=2021, help='seed for random and torch')
args = parser.parse_args()
print(args)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    setseed(args.seed)
    print('Loading data...')
    train = pd.read_csv(args.data_dir + 'train_0_clean_part_aug_by_trans.tsv', sep = '\t')
    valid = pd.read_csv(args.data_dir + 'valid_0_clean.tsv', sep = '\t')

    train['Description'] = train['Description'].fillna('')
    valid['Description'] = valid['Description'].fillna('')
    sub_columns = ['Title', 'Description', 'Image_id', 'Product_id', 'Prdtypecode']

    train_data = ECTextDataset(train[sub_columns].values, args.text_ptm_dir)
    valid_data = ECTextDataset(valid[sub_columns].values, args.text_ptm_dir)

    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, num_workers = 8)

    model = TextOnly(args.text_ptm_dir, args.text_ft_dim, args.num_classes).to(device)
    model = nn.DataParallel(model, device_ids=[2, 3])

    if args.test:
        ckpt = torch.load(args.result_dir + '/best_checkpoint_bertpooled_seed2021_bs128_lr5e-05_ep30_numiter17910_warmup1791.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        acc, macro_f1, all_preds = validate(valid_loader, model)
        print('Validation Best Results: accuracy: {:.4f}, macro f1: {:.4f}'.format(acc, macro_f1))
        valid['preds'] = all_preds
        valid.to_csv(args.result_dir + '/valid_0_preds_camembert_base.tsv', index = False, sep = '\t')
        return
    
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma = 3)
    # optimizer = optim.Adam(model.parameters(), args.lr)
    # scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)
    optimizer = AdamW(model.parameters(), args.lr)
    num_training_steps = int(train.shape[0] / args.batch_size) * args.epoch
    num_warmup_steps = int(num_training_steps * 0.1)
    # print(train.shape[0], num_training_steps, num_warmup_steps)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_training_steps, num_cycles = 6)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_training_steps)

    best_f1 = 0
    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        trainForEpoch(train_loader, model, optimizer, scheduler, epoch, args.epoch, criterion, log_aggr = 200)
        acc, macro_f1, all_preds = validate(valid_loader, model)

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        torch.save(ckpt_dict, args.result_dir + '/latest_checkpoint.pth.tar')

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(ckpt_dict, args.result_dir + '/best_checkpoint_bertpooled_camemlarge_part_aug_trans_seed{}_bs{}_lr{}_ep{}_numiter{}_warmup{}.pth.tar'.format(
                                                    args.seed, args.batch_size, args.lr, args.epoch, num_training_steps, num_warmup_steps))

        print('Epoch {} validation: accuracy: {:.4f}, macro f1: {:.4f}, best macro f1: {:.4f}'.format(epoch, acc, macro_f1, best_f1))


def trainForEpoch(train_loader, model, optimizer, scheduler, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (input_ids, attention_mask, token_type_ids, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(input_ids) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model):
    model.eval()

    all_preds, all_labels = [], [] 
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in tqdm(valid_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask, token_type_ids)
            logits = F.softmax(outputs, dim = 1)
            preds = torch.argmax(logits, dim = 1)
            all_preds.extend(preds.data.cpu().numpy())
            all_labels.extend(labels.data.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average = 'macro')
    return acc, macro_f1, all_preds


if __name__ == '__main__':
    main()
