#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on May 30, 2020
Updated on June 18 2020
@author: wangshuo
@desc: train script with image modal for SIGIR 2020 challenge Task 1.

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

from model import UniImageRes152
from dataloader import ECImageDataset
from utils import LabelSmoothingLoss

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/competition/SIGIR2020/Task_1/data/preprocessed/', help='preprocessed dataset directory')
parser.add_argument('--img_dir', default='/home/competition/SIGIR2020/Raw_data/image/', help='image dataset directory')
parser.add_argument('--img_ptm_path', default='/home/competition/SIGIR2020/pretrained/resnet152-b121ed2d.pth', help='pretrained image model path')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--num_classes', type=int, default=27, help='the number of classes')
parser.add_argument('--img_ft_dim', type=int, default=2048, help='the dim of image features')
parser.add_argument('--result_dir', default='/home/competition/SIGIR2020/Task_1/imgmodal/results', help='directory to store results')
parser.add_argument('--epoch', type=int, default=60, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=10, help='the number of steps after which the learning rate decay')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--seed', type=int, default=1021, help='seed for random and torch')
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    train = pd.read_csv(args.data_dir + 'train_0_clean.tsv', sep = '\t')
    valid = pd.read_csv(args.data_dir + 'valid_0_clean.tsv', sep = '\t')

    train['Description'] = train['Description'].fillna('')
    valid['Description'] = valid['Description'].fillna('')
    sub_columns = ['Title', 'Description', 'Image_id', 'Product_id', 'Prdtypecode']
    
    train_img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    valid_img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
 
    train_img_dir = os.path.join(args.img_dir, 'image_training') 
    train_data = ECImageDataset(train[sub_columns].values, train_img_dir, img_transform = train_img_transform)
    valid_data = ECImageDataset(valid[sub_columns].values, train_img_dir, img_transform = valid_img_transform)

    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, num_workers = 8)

    model = UniImageRes152(args.img_ptm_path, args.img_ft_dim, args.num_classes).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1])

    if args.test:
        ckpt = torch.load(args.result_dir + '/best_checkpoint_seed1021_bs128_lr0.01_ep60.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        acc, macro_f1, all_preds = validate(valid_loader, model)
        print('Validation Best Results: accuracy: {:.4f}, macro f1: {:.4f}'.format(acc, macro_f1))
        # valid['preds'] = all_preds
        # valid.to_csv(args.result_dir + '/valid_0_preds.tsv', index = False, sep = '\t')
        return
    
    # optimizer = optim.Adam(model.parameters(), args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingLoss(0.1, args.num_classes)
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    best_f1 = 0
    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 50)
        scheduler.step(epoch = epoch)
        acc, macro_f1, all_preds = validate(valid_loader, model)

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if not os.path.exists('results'):
            os.makedirs('results')

        torch.save(ckpt_dict, args.result_dir + '/latest_checkpoint.pth.tar')

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(ckpt_dict, args.result_dir + '/best_checkpoint_seed{}_bs{}_lr{}_ep{}.pth.tar'.format(args.seed, args.batch_size, args.lr, args.epoch))

        print('Epoch {} validation: accuracy: {:.4f}, macro f1: {:.4f}, best macro f1: {:.4f}'.format(epoch, acc, macro_f1, best_f1))


def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()
    sum_epoch_loss = 0

    start = time.time()
    for i, (imgs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(imgs) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model):
    model.eval()
    all_preds, all_labels = [], [] 
    with torch.no_grad():
        for imgs, labels in tqdm(valid_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            logits = F.softmax(outputs, dim = 1)
            preds = torch.argmax(logits, dim = 1)
            all_preds.extend(preds.data.cpu().numpy())
            all_labels.extend(labels.data.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average = 'macro')
    return acc, macro_f1, all_preds


if __name__ == '__main__':
    main()
