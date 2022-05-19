#!/usr/bin/env python
# coding: utf-8

import os
import io
import sys
from os import listdir
from os.path import isfile,isdir,join,basename
import random
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.models import load_model
from openpyxl.workbook import Workbook
from cnfg import Cnfg

from BS_AlignForm import align, cropped_from_aligned
from BS_TextExtractor import FindText
from BS_Checkbox import checkbox_predict


class BlueShield:
    def __init__(self, template_type):
        p = Cnfg()
        base_folder = p.base_folder
        temp_folder = p.temp_folder

        if template_type == 'brc_template1':
            template_file = p.template_file1
            address_pos = p.address_pos1
            barcode_pos = p.barcode_pos1
            email_pos = p.email_pos1
            phone_pos = p.phone_pos1
            checkbox_pos = p.checkbox_pos1

        pos_dict = {'address': address_pos, 
                    'barcode': barcode_pos,
                    'email': email_pos,
                    'phone': phone_pos,
                    'checkbox': checkbox_pos}

        canvas = np.full(p.canvas_shape, 255, dtype=np.uint8)

        checkbox_model = load_model('checkbox_model.h5')
        # eraseline_model = load_model('erase_lines.h5')

        self.base_folder = base_folder
        self.temp_folder = temp_folder
        self.template_file = template_file
        self.pos_dict = pos_dict
        self.canvas = canvas
        self.checkbox_model = checkbox_model
        # self.eraseline_model = eraseline_model 

    def extract_data(self, file_path):
        template_file = self.template_file
        canvas = self.canvas
        checkbox_model = self.checkbox_model
        #eraseline_model = self.eraseline_model

        # Align the source images with template
        template = cv2.imread(str(template_file), 1)
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        matrix = align(template, img)
        res = cv2.warpPerspective(img, matrix, (template.shape[1], template.shape[0]))
        sections = cropped_from_aligned(res, self.pos_dict)
        img_path = {}
        for kind in ['address', 'barcode', 'email', 'phone', 'checkbox']:
            img_path[kind] = join(self.temp_folder, kind, kind + '_' + basename(file_path))
            cv2.imwrite(img_path[kind], sections[kind])
        
        # Extract textual data from images
        result_dict = {}
        offset_h, offset_w = 0, 0
        for kind in ['address', 'phone', 'email']:
            img1 = sections[kind]
            h, w = img1.shape
            canvas[offset_h : offset_h+h, offset_w : offset_w+w] = img1
            offset_h = offset_h + h + 100
        path = join(self.temp_folder, 'canvas', 'canvas' + '_' + basename(file_path))
        print(path)
        cv2.imwrite(path, canvas)

        obj = FindText(path)
        result_dict = obj.extract_address()
        result_dict['phone'] = obj.extract_phone()
        result_dict['email'] = obj.extract_email()

        # Extract Checkbox data
        checkbox_path = img_path['checkbox']
        pred = checkbox_predict(checkbox_path, checkbox_model)
        values = np.round(np.clip(pred, 0, 1)).astype(int) 
        lst = [f for sublist in values for f in sublist]
        i = 0
        for box in ['mail_checkbox', 'phone_checkbox', 'email_checkbox']:
            result_dict[box] = lst[i]
            i += 1

        return result_dict
            
    def extract_data_from_folder(self, src):
        base_folder = self.base_folder
        template_file = self.template_file
        checkbox_model = self.checkbox_model
        #eraseline_model = self.eraseline_model

        # Align the source images with template
        template = cv2.imread(template_file, 1)
        files = os.listdir(src)
        for f in files:
            img = cv2.imread(join(src, f), cv2.IMREAD_GRAYSCALE)
            matrix = align(template, img)
            res = cv2.warpPerspective(img, matrix, (template.shape[1], template.shape[0]))
            sections = cropped_from_aligned(res, self.pos_dict)
            for kind in ['address', 'barcode', 'email', 'phone', 'checkbox']:
                cv2.imwrite(join(base_folder, kind, kind + '_' + f), sections[kind])

        # Extract textual data from images
        df_text = pd.DataFrame(columns = ['address', 'phone', 'email'])
        for kind in ['address', 'phone', 'email']:
            data = []
            path = join(base_folder, kind)
            img_files = os.listdir(path)
            img_path = [join(path, f) for f in img_files if f.endswith('.png')]
            for img in img_path:
                obj = FindText(img)
                text = obj.document.text
                data.append(text)
            df_text[kind] = data

        # Extract Checkbox
        df_check = pd.DataFrame(columns = ['mail_checkbox', 'phone_checkbox', 'email_checkbox'])
        checkbox_path = join(base_folder, 'checkbox')
        img_files = os.listdir(checkbox_path)
        img_path = [join(checkbox_path, f) for f in img_files if f.endswith('.png')]

        for img in img_path:
            pred = checkbox_predict(img, checkbox_model)
            values = np.round(np.clip(pred, 0, 1)).astype(int) 
            lst = [f for sublist in values for f in sublist]
            df_check.loc[len(df_check)] = lst

        result = pd.concat([df_text, df_check], axis = 1)
        # writing to Excel
        data = pd.ExcelWriter(join(base_folder, 'output', 'output.xlsx'))
        result.to_excel(data)
        data.save()

    def save_result(self):
        pass

if __name__ == '__main__':
    docpro = BlueShield('brc_template1')
    file = r'C:\Users\vinee\source\repos\BlueShield_DocPro\filled_forms\8_filled.png'
    data = docpro.extract_data(file)
    print(data)