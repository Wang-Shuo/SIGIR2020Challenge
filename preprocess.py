#!/usr/bin/env python37
# -*- coding: utf-8 -*-
import os
import re
import json
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

def train_valid_split():
    ''' split all annotated samples into train and validation set
    '''
    data_root = 'data/raw/'
    X_train = pd.read_csv(data_root + 'phase1/data/X_train.tsv', sep = '\t')
    Y_train = pd.read_csv(data_root + 'phase1/data/Y_train.tsv', sep = '\t')
    train = X_train.merge(Y_train[['Integer_id', 'Prdtypecode']], how = 'left', on = 'Integer_id')
    
    # 将原始类别重编码
    class_idx = 0
    code2idx = {}
    idx2code = {}
    for prd_code in train['Prdtypecode'].unique():
        code2idx[prd_code] = class_idx
        idx2code[class_idx] = prd_code
        class_idx += 1

    train['Prdtypecode'] = train['Prdtypecode'].map(code2idx)
    if not os.path.exists('data/preprocessed'):
        os.makedirs('data/preprocessed')

    train.to_csv('data/preprocessed/train.tsv', sep ='\t', index = False)
    with open('data/preprocessed/code2idx.json', 'w') as f:
        json.dump(str(code2idx), f)
    with open('data/preprocessed/idx2code.json', 'w') as f:
        json.dump(str(idx2code), f)

    # 划分训练集和验证集
    X = train['Integer_id'].values
    Y = train['Prdtypecode'].values

    valid_size = 0.1
    sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=2020)

    for i, (train_index, valid_index) in enumerate(sss.split(X, Y)):
        train_integer_ids, valid_integer_ids = X[train_index], X[valid_index]
        train_df = train[train['Integer_id'].isin(train_integer_ids)]
        valid_df = train[train['Integer_id'].isin(valid_integer_ids)]
        train_df.to_csv('data/preprocessed/train_{}.tsv'.format(i), sep = '\t', index = False)
        valid_df.to_csv('data/preprocessed/valid_{}.tsv'.format(i), sep = '\t', index = False)


def text_clean():
    '''text clean for product titles and descriptions
    '''
    def cleanhtml(raw_html):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, ' ', raw_html)
        cleansp = re.sub('\s+', ' ', cleantext)
        return cleansp

    # text clean for train and valid set
    data_root = 'data/raw/'
    train = pd.read_csv('data/preprocessed/train_0.tsv', sep = '\t')
    valid = pd.read_csv('data/preprocessed/valid_0.tsv', sep = '\t')
    train['Title'] = train['Title'].astype(str).apply(cleanhtml)
    train['Description'] = train['Description'].astype(str).apply(cleanhtml)
    valid['Title'] = valid['Title'].astype(str).apply(cleanhtml)
    valid['Description'] = valid['Description'].astype(str).apply(cleanhtml)
    train.to_csv('data/preprocessed/train_0_clean.tsv', index = False, sep = '\t')
    valid.to_csv('data/preprocessed/valid_0_clean.tsv', index = False, sep = '\t')

    # text clean for phase1 test set
    X_test = pd.read_csv(data_root + 'phase1/data/x_test_task1_phase1.tsv', sep = '\t')
    X_test['Title'] = X_test['Title'].astype(str).apply(cleanhtml)
    X_test['Description'] = X_test['Description'].astype(str).apply(cleanhtml)
    X_test.to_csv('data/preprocessed/x_test_task1_phase1_clean.tsv', index = False, sep = '\t')

    # text clean for phase2 test set
    X_test = pd.read_csv(data_root + 'phase2/x_test_task1_phase2.tsv', sep = '\t')
    X_test['Title'] = X_test['Title'].astype(str).apply(cleanhtml)
    X_test['Description'] = X_test['Description'].astype(str).apply(cleanhtml)
    X_test.to_csv('data/preprocessed/x_test_task1_phase2_clean.tsv', index = False, sep = '\t')


def cv_split():
    ''' k-fold split for computing predicted probabilities for all training samples. 
    '''
    train = pd.read_csv('data/preprocessed/train.tsv', sep = '\t')
    X = train['Integer_id'].values
    Y = train['Prdtypecode'].values

    sss = StratifiedKFold(n_splits=4)

    for i, (train_index, valid_index) in enumerate(sss.split(X, Y)):
        train_integer_ids, valid_integer_ids = X[train_index], X[valid_index]
        train_df = train[train['Integer_id'].isin(train_integer_ids)]
        valid_df = train[train['Integer_id'].isin(valid_integer_ids)]
        train_df.to_csv('data/preprocessed/cv/train_cv_{}.tsv'.format(i), sep = '\t', index = False)
        valid_df.to_csv('data/preprocessed/cv/valid_cv_{}.tsv'.format(i), sep = '\t', index = False)


if __name__ == "__main__":
    train_valid_split()
    text_clean()
    cv_split()