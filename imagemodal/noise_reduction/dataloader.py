#!/usr/bin/env python37
# -*- coding: utf-8 -*-
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class ECImageDataset(Dataset):
	def __init__(self, data_list, img_dir, img_transform = None):
		self.data_list = data_list
		self.img_dir = img_dir
		self.img_transform = img_transform

	def __getitem__(self, index):
		title, desc, img_id, product_id, product_code = self.data_list[index]
		# image
		img_path = os.path.join(self.img_dir, 'image_{}_product_{}.jpg'.format(img_id, product_id))
		img = Image.open(img_path)
		if self.img_transform:
			img = self.img_transform(img)
        
		return img, product_code 

	def __len__(self):
		return len(self.data_list)
    