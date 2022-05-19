#!/usr/bin/env python
# coding: utf-8


import os
import io
import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from enum import Enum
from google.cloud import vision
from typing import Tuple


# In[3]:

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


class FindText():

    def __init__(self, image_path):

        # calling up google vision json file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
   
        # initiate a client
        client = vision.ImageAnnotatorClient()

        # load image into memory
        with io.open(image_path, 'rb') as image_file:
            file_content = image_file.read()

        # perform text detection from the image
        image_detail = vision.Image(content=file_content)
        response = client.document_text_detection(image=image_detail, image_context={"language_hints": ["en"]})
        document = response.full_text_annotation
        
        """Returns document bounds given an image."""
        bounds=[]
        feature = FeatureType.BLOCK
        
        for page in document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            if feature == FeatureType.SYMBOL:
                                bounds.append(symbol.bounding_box)

                        if feature == FeatureType.WORD:
                            bounds.append(word.bounding_box)

                    if feature == FeatureType.PARA:
                        bounds.append(paragraph.bounding_box)

                if feature == FeatureType.BLOCK:
                    bounds.append(block.bounding_box)

        # The list 'bounds' contains the coordinates of the bounding boxes.
        self.bounds = bounds
        self.document = document

    @staticmethod
    def vertices_to_box(v1, v2, v3, v4) -> Tuple: #[left, right, bot, top]
        xs = [i.x for i in [v1, v2, v3, v4]]
        ys = [i.y for i in [v1, v2, v3, v4]]
        left = min(xs)
        right = max(xs)
        bot = min(ys)
        top = max(ys)
        return left, right, bot, top

    def extract_address(self, flag=0):
        bounds = self.bounds
        ID = ''
        name = ''
        address = ''
        data = {}
        for page in self.document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            min_x, max_x, min_y, max_y = self.vertices_to_box(symbol.bounding_box.vertices[0],
                                                                              symbol.bounding_box.vertices[1],
                                                                              symbol.bounding_box.vertices[2],
                                                                              symbol.bounding_box.vertices[3])
                            
                            if(min_x >= bounds[flag].vertices[0].x-20 and max_x <= bounds[flag].vertices[2].x+20 and min_y >= bounds[flag].vertices[0].y-10 and max_y <= bounds[flag].vertices[0].y+40):
                                ID += symbol.text
                            data['ID'] = ID
                            
                            if(min_x >= bounds[flag].vertices[0].x-20 and max_x <= bounds[flag].vertices[2].x+20 and min_y >= bounds[flag].vertices[0].y+30 and max_y <= bounds[flag].vertices[0].y+90):
                                name += symbol.text
                                if(symbol.property.detected_break.type==1):
                                    name += ' '
    #                             if(symbol.property.detected_break.type==2):
    #                                 name += '\t'
    #                             if(symbol.property.detected_break.type==3 or symbol.property.detected_break.type==5):
    #                                 name += '\n'
                            data['name'] = name
                            
                            if(min_x >= bounds[flag].vertices[0].x-20 and max_x <= bounds[flag].vertices[2].x+20 and min_y >= bounds[flag].vertices[0].y+70 and max_y <= bounds[flag].vertices[2].y+20):
                                address += symbol.text
                                if(symbol.property.detected_break.type==1):
                                    address += ' '
    #                             if(symbol.property.detected_break.type==2):
    #                                 address += '\t'
    #                             if(symbol.property.detected_break.type==3 or symbol.property.detected_break.type==5):
    #                                 address += '\n'
                            data['address'] = address
                            
        return data

    def extract_phone(self):
        bounds = self.bounds
        text = ''
        for page in self.document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            min_x, max_x, min_y, max_y = self.vertices_to_box(symbol.bounding_box.vertices[0],
                                                                              symbol.bounding_box.vertices[1],
                                                                              symbol.bounding_box.vertices[2],
                                                                              symbol.bounding_box.vertices[3])

                            if(min_x >= bounds[-1].vertices[0].x-20 and max_x <= bounds[-1].vertices[2].x+20 and min_y >= bounds[0].vertices[2].y+20 and max_y <= bounds[-1].vertices[0].y-20):
                                text += symbol.text
        return text

    def extract_email(self, flag=-1):
        bounds = self.bounds
        text = ''
        for page in self.document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            min_x=min(symbol.bounding_box.vertices[0].x,symbol.bounding_box.vertices[1].x,symbol.bounding_box.vertices[2].x,symbol.bounding_box.vertices[3].x)
                            max_x=max(symbol.bounding_box.vertices[0].x,symbol.bounding_box.vertices[1].x,symbol.bounding_box.vertices[2].x,symbol.bounding_box.vertices[3].x)
                            min_y=min(symbol.bounding_box.vertices[0].y,symbol.bounding_box.vertices[1].y,symbol.bounding_box.vertices[2].y,symbol.bounding_box.vertices[3].y)
                            max_y=max(symbol.bounding_box.vertices[0].y,symbol.bounding_box.vertices[1].y,symbol.bounding_box.vertices[2].y,symbol.bounding_box.vertices[3].y)
                            if(min_x >= bounds[flag].vertices[0].x-20 and max_x <= bounds[flag].vertices[2].x+20 and min_y >= bounds[flag].vertices[0].y-20 and max_y <= bounds[flag].vertices[2].y+20):
                                text += symbol.text
        return text


if __name__ == '__main__':
    img_path = r'C:\Users\vinee\source\repos\BlueShield_DocPro\filled_forms\1_filled.png'
    find_text = FindText(img_path)
    find_text.extract_phone()




