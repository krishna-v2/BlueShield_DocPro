
import os
import io
import sys
from os import listdir
from os.path import isfile,isdir,join,basename, dirname
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
from BlueShieldDocPro.cnfg import Cnfg

from BlueShieldDocPro.BS_AlignForm import align, cropped_from_aligned
from BlueShieldDocPro.BS_TextExtractor import find_text
from BlueShieldDocPro.BS_Checkbox import checkbox_predict
from BlueShieldDocPro.get_email import extract_email

class BlueShield:
    def __init__(self, template_type):
        p = Cnfg()
        base_folder = p.base_folder
        temp_folder = p.temp_folder

        if template_type == 'brc_template1':
            pos_dict = p.pos
            template_file = p.template_file1
        else:
            raise ValueError("Template not supported")

        canvas = np.full(p.canvas_shape, 255, dtype=np.uint8)

        checkbox_model = load_model(join(dirname(__file__), 'checkbox_model.h5'))
        # eraseline_model = load_model('erase_lines.h5')

        self.base_folder = base_folder
        self.temp_folder = temp_folder
        self.template_file = template_file
        self.pos_dict = pos_dict
        self.canvas = canvas
        self.checkbox_model = checkbox_model
        # self.eraseline_model = eraseline_model 

    def extract_data(self, file_path):
        p = Cnfg()
        all_kinds = p.all_kinds
        text_kinds = p.text_kinds
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
        for kind in all_kinds:
            img_path[kind] = join(self.temp_folder, kind, kind + '_' + basename(file_path))
            cv2.imwrite(img_path[kind], sections[kind])
        
        # Extract textual data from images
        result_dict = {}
        for kind in text_kinds:
            offset_h, offset_w = p.pos[kind]['canvas_h_offset'], 0
            img1 = sections[kind]
            h, w = img1.shape
            canvas[offset_h : offset_h+h, offset_w : offset_w+w] = img1
        path = join(self.temp_folder, 'canvas', 'canvas' + '_' + basename(file_path))

        cv2.imwrite(path, canvas)

        result_dict = find_text(path)

        # Extract Checkbox data
        checkbox_path = img_path['checkbox']
        pred = checkbox_predict(checkbox_path, checkbox_model)
        values = np.round(np.clip(pred, 0, 1)).astype(int) 
        lst = [f for sublist in values for f in sublist]
        i = 0
        for box in ['mail_checkbox', 'phone_checkbox', 'email_checkbox']:
            result_dict[box] = lst[i]
            i += 1

        email, count_score, char_scores = extract_email(self.canvas)
        result_dict['email_canvas'] = {'email': email, 'count_score': count_score, 'char_scores': char_scores}

        return result_dict

if __name__ == '__main__':
    docpro = BlueShield('brc_template1')
    file = r'C:\Users\vinee\source\repos\BlueShield_DocPro\filled_forms\12.png'
    data = docpro.extract_data(file)
    print(data)
