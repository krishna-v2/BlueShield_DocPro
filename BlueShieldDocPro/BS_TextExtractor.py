#!/usr/bin/env python
# coding: utf-8

import sys
from os.path import dirname
sys.path.append(dirname(__file__))

import os
import io
import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from enum import Enum
from google.cloud import vision
from typing import Tuple
from BlueShieldDocPro.cnfg import Cnfg
import tensorflow as tf
import numpy as np


# In[3]:

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def find_text(image_path, document_only=False):
    # calling up google vision json file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'neutron-345122-39e5256f7765.json'

    # initiate a client
    client = vision.ImageAnnotatorClient()

    # load image into memory
    with io.open(image_path, 'rb') as image_file:
        file_content = image_file.read()

    # perform text detection from the image
    image_detail = vision.Image(content=file_content)
    response = client.document_text_detection(image=image_detail, image_context={"language_hints": ["en"]})
    document = response.full_text_annotation
    if document_only:
        return document

    return extract_text_helper(document)


def intersect(tl, tr, br, bl, canvs_pos):
    xs = [tl.x, tr.x, br.x, bl.x]
    ys = [tl.y, tl.y, br.y, bl.y]

    lw = min(xs) # left word found by vision
    rw = max(xs)
    tw = min(ys)
    bw = max(ys)

    lw = max(0, lw)
    tw = max(0, tw)

    lc = canvs_pos['canvas_w_offset'] # left of box in canvas
    rc = lc + canvs_pos['we'] - canvs_pos['ws']
    tc = canvs_pos['canvas_h_offset']
    bc = tc + canvs_pos['he'] - canvs_pos['hs']

    # return 0 if there is no overlap
    if lw == rw or tw == bw:
        return 0
    if lw > rc or lc > rw:
        return 0
    if tw > bc or tc > bw:
        return 0

    area_overlap = (max(lw, lc) - min(rw, rc)) * (max(tw, tc) - min(bw, bc))
    area_word = (rw - lw) * (bw - tw)

    return area_overlap / area_word

def check_list_overlap(words):
    if len(words) < 2:
        return words

    boxes = []
    scores = []
    for word in words:
        v = [word.bounding_box.vertices[i] for i in range(4)]
        boxes.append([v[0].y, v[0].x, v[2].y, v[2].x])
        scores.append(word.confidence)

    boxes = np.array(boxes)
    scores = np.array(scores)

    indexes = tf.image.non_max_suppression(boxes, scores, 100, iou_threshold=0.2)
    indexes = sorted(indexes)
    words = [words[i] for i in indexes]

    return words

def extract_text_helper(document):
    p = Cnfg()
    canvas_pos = p.pos
    kinds = p.text_kinds

    # Collect words of high overlap with canvas positions of each kind
    words = {i: [] for i in kinds}
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    v = [word.bounding_box.vertices[i] for i in range(4)]
                    for kind in kinds:
                        overlap = intersect(v[0], v[1], v[2], v[3], canvas_pos[kind])
                        if overlap > 0.5:
                            words[kind].append(word)

    for kind in kinds:
        words[kind] = check_list_overlap(words[kind])

    # Collect symbols from each word found
    text = {i: [] for i in kinds}
    for kind in kinds:
        for word in words[kind]:
            for symbol in word.symbols:
                if kind == 'phone':
                    v = [symbol.bounding_box.vertices[i] for i in range(4)]
                    overlap = intersect(v[0], v[1], v[2], v[3], canvas_pos['left_p'])
                    if overlap > 0.5:
                        continue
                    overlap = intersect(v[0], v[1], v[2], v[3], canvas_pos['right_p'])
                    if overlap > 0.5:
                        continue

                text[kind].append(symbol.text)
                if symbol.property.detected_break.type == 1:
                    text[kind].append(' ')
                if symbol.property.detected_break.type in [3, 5]:
                    text[kind].append('\n')

    # concatenate symbol list
    for kind in text.keys():
        text[kind] = ''.join(text[kind])

    address_lines = text['address'].split('\n')
    if len(address_lines) > 2:
        text['ID'] = address_lines.pop(0)
        text['name'] = address_lines.pop(0)
        text['address'] = '\n'.join(address_lines)
    else:
        raise ValueError("Not enough lines detected in address")

    text['phone'] = ''.join(e for e in text['phone'] if e.isdigit())

    return text


if __name__ == '__main__':
    img_path = r'C:\Users\vinee\source\repos\BlueShield_DocPro\temp\canvas\canvas_18.png'
    text = find_text(img_path)
    print(text)
