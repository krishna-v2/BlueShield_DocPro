#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, isfile
import os


def align(template, img):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template,None)
    kp2, des2 = sift.detectAndCompute(img,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good_points = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good_points.append(m)

    query_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    train_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)

#     h, w = template.shape[:2]
#     pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
#     dst_trans = cv2.perspectiveTransform(pts, matrix)
#     homography = cv2.polylines(img, [np.int32(dst_trans)], True, (0, 0, 0), 10)
    return matrix

def crop(form_files, anchor_file, src, dst):
    anchor = cv2.imread(anchor_file)
    h, w = anchor.shape[:2]
    for file in form_files:
        form = cv2.imread(join(src, file))
        matrix = align(anchor, form)
        cropped_img = cv2.warpPerspective(form[:,:,0], matrix, (w,h))
        cv2.imwrite(join(dst, file), cropped_img)

def cropped_from_aligned(img, pos_dict):
    sections = {}
    for kind in ['address', 'barcode', 'email', 'phone', 'checkbox']:
        d = pos_dict[kind]
        sections[kind] = img[d['hs']:d['he'], d['ws']:d['we']]
    return sections
