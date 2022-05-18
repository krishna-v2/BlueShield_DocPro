#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tqdm import tqdm


class BatchGen(tf.keras.utils.Sequence):
    def __init__(self, 
                 data_folder,
                 data_df,
                 kind='train', 
                 samples=1000,
                 batch_size=32,
                 img_wh=(32,32)):
        if kind == 'train':
            df = data.iloc[:int(0.8*len(data)), :]
        else:
            df = data.iloc[int(0.8*len(data)):, :]
        checkboxes = get_rnd_checkboxes(samples, data_folder, df)
        self.imgs = np.array([cv2.resize(item[0], img_wh) for item in checkboxes])
        self.labels = np.array([item[1] for item in checkboxes])
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.labels)//self.batch_size
    
    def __getitem__(self, index):
        s = index * self.batch_size
        e = s + self.batch_size
        
        return self.imgs[s:e, ...], self.labels[s:e]
        
def get_rnd_checkboxes(count, data_folder, data_df):
    checkboxes = []
    for i in tqdm(range(count)):
        row = data.sample(1).iloc[0]
        img_path = data_folder.joinpath(row['rel_path'])
        img = cv2.imread(str(img_path))
        for j in range(3):
            l_rnd = int(l * (1 + l_tol * np.random.rand()))
            h_s = c[j][0] - l_rnd//2
            h_e = h_s + l_rnd
            w_s = c[j][1] - l_rnd//2
            w_e = w_s + l_rnd
            label = row[col_names[j]]
            checkboxes.append((img[h_s:h_e, w_s:w_e, :], label))
    return checkboxes

def checkbox_predict(img, model):
    
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = np.dstack([img]*3)
    img = cv2.resize(img,(32, 96))
    img = img.astype(np.float32)

    t = img[:32, :, :]
    m = img[32:64, :, :]
    b = img[64:, :, :]
    img = np.array([t, m, b])
    
    prediction = model.predict(img)
    return prediction







