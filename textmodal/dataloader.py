#!/usr/bin/env python37
# -*- coding: utf-8 -*-
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.tokenization_camembert import CamembertTokenizer 


class ECTextDataset(Dataset):
	def __init__(self, data_list, text_ptm_dir):
		self.data_list = data_list
		self.tokenizer = CamembertTokenizer.from_pretrained(text_ptm_dir)

	def __getitem__(self, index):
		title, desc, img_id, product_id, product_code = self.data_list[index]
		# text
		if desc == '':
			desc = self.tokenizer.pad_token
		tokenizer_outputs = self.tokenizer.encode_plus(title, desc, add_special_tokens = True, return_token_type_ids = True, 
														pad_to_max_length = True, max_length = 128)
		input_ids = tokenizer_outputs['input_ids']
		attention_mask = tokenizer_outputs['attention_mask']
		token_type_ids = tokenizer_outputs['token_type_ids']
        
		return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(token_type_ids), product_code 

	def __len__(self):
		return len(self.data_list)


class ECTextTestDataset(Dataset):
	def __init__(self, data_list, text_ptm_dir):
		self.data_list = data_list
		self.tokenizer = CamembertTokenizer.from_pretrained(text_ptm_dir)

	def __getitem__(self, index):
		title, desc, img_id, product_id = self.data_list[index]
		# text
		if desc == '':
			desc = self.tokenizer.pad_token
		tokenizer_outputs = self.tokenizer.encode_plus(title, desc, add_special_tokens = True, return_token_type_ids = True, 
														pad_to_max_length = True, max_length = 128)
		input_ids = tokenizer_outputs['input_ids']
		attention_mask = tokenizer_outputs['attention_mask']
		token_type_ids = tokenizer_outputs['token_type_ids']
        
		return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(token_type_ids)

	def __len__(self):
		return len(self.data_list)


class ECTitleDataset(Dataset):
	def __init__(self, data_list, text_ptm_dir):
		self.data_list = data_list
		self.tokenizer = CamembertTokenizer.from_pretrained(text_ptm_dir)

	def __getitem__(self, index):
		title, desc, img_id, product_id, product_code = self.data_list[index]
		tokenizer_outputs = self.tokenizer.encode_plus(title, add_special_tokens = True, 
							  pad_to_max_length = True, max_length = 32)
		input_ids = tokenizer_outputs['input_ids']
		attention_mask = tokenizer_outputs['attention_mask']
		# token_type_ids = tokenizer_outputs['token_type_ids']
        
		return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), product_code 

	def __len__(self):
		return len(self.data_list)


class ECDescDataset(Dataset):
	def __init__(self, data_list, text_ptm_dir):
		self.data_list = data_list
		self.tokenizer = CamembertTokenizer.from_pretrained(text_ptm_dir)

	def __getitem__(self, index):
		title, desc, img_id, product_id, product_code = self.data_list[index]
		if desc == '':
			desc = self.tokenizer.pad_token
		tokenizer_outputs = self.tokenizer.encode_plus(desc, add_special_tokens = True, 
							  pad_to_max_length = True, max_length = 128)
		input_ids = tokenizer_outputs['input_ids']
		attention_mask = tokenizer_outputs['attention_mask']
		# token_type_ids = tokenizer_outputs['token_type_ids']
        
		return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), product_code 

	def __len__(self):
		return len(self.data_list)
