import matplotlib.pyplot as plt
from cnfg import Cnfg, EmailP
import cv2
import numpy as np
import tensorflow as tf
from BlueShieldDocPro.BS_TextExtractor import find_text, intersect
import collections
from os.path import join


def extract_email(docpro_canvas):
    docpro_cnfg = Cnfg()
    email_cnfg = EmailP()

    email_pos = docpro_cnfg.pos['email']
    pcts = email_cnfg.pcts
    npcts = len(pcts)
    save_folder = email_cnfg.save_folder

    spacing = int((email_pos['he'] - email_pos['hs']) * max(pcts) * 0.01)
    canvas_h = (2 * npcts - 1) * spacing + 1
    canwas_w = int((email_pos['we'] - email_pos['ws']) * max(pcts) * 0.01 + 1)

    email_img = docpro_canvas[email_pos['canvas_h_offset']: email_pos['canvas_h_offset'] + email_pos['he'] - email_pos['hs'],
          email_pos['canvas_w_offset']: email_pos['canvas_w_offset'] + email_pos['we'] - email_pos['ws']]

    pos_dict = save_canvases(email_img, canvas_h, canwas_w, pcts, spacing, save_folder)

    doc_canvas = find_text(join(save_folder, 'canvas.png'), True)
    doc_canvas_binary = find_text(join(save_folder, 'canvas_binary.png'), True)
    doc_canvas_tozero = find_text(join(save_folder, 'canvas_tozero.png'), True)

    symbols_canvas = doc_to_symbols(doc_canvas, pos_dict, pcts)
    symbols_canvas_binary = doc_to_symbols(doc_canvas_binary, pos_dict, pcts)
    symbols_canvas_tozero = doc_to_symbols(doc_canvas_tozero, pos_dict, pcts)

    symbols = list(symbols_canvas.values()) + list(symbols_canvas_binary.values()) + list(symbols_canvas_tozero.values())

    frequent1, frequent2, leftovers = find_top_2_symbols_by_len(symbols)

    email_str, count_score, char_scores = find_common_str(frequent1)

    return email_str, count_score, char_scores


def save_canvases(email_img, canvas_h, canwas_w, pcts, spacing, save_folder):
    canvas = np.full((canvas_h, canwas_w), 255, dtype=np.uint8)
    canvas_binary = canvas.copy()
    canvas_tozero = canvas.copy()

    pos_dict = {}
    for i, pct in enumerate(pcts):
        resized = cv2.resize(email_img, None, fx=pct * 0.01, fy=pct * 0.01, interpolation=cv2.INTER_AREA)

        th_binary = resized.copy()
        ret3, th_binary = cv2.threshold(th_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        th_tozero = resized.copy()
        ret3, th_tozero = cv2.threshold(th_tozero, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

        hs = i * spacing * 2
        he = hs + resized.shape[0]
        ws = 0
        we = resized.shape[1]
        canvas[hs:he, ws:we] = resized
        canvas_binary[hs:he, ws:we] = th_binary
        canvas_tozero[hs:he, ws:we] = th_tozero
        pos_dict[pct] = {'hs': 0, 'he': he - hs, 'ws': 0, 'we': we - ws, 'canvas_h_offset': hs, 'canvas_w_offset': ws}

    cv2.imwrite(join(save_folder, 'canvas.png'), canvas)
    cv2.imwrite(join(save_folder, 'canvas_binary.png'), canvas_binary)
    cv2.imwrite(join(save_folder, 'canvas_tozero.png'), canvas_tozero)

    return pos_dict


def doc_to_symbols(document, pos, pcts):
    symbols = {pct: [] for pct in pcts}
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                v1 = [paragraph.bounding_box.vertices[i] for i in range(4)]
                for pct in pcts:
                    overlap = intersect(v1[0], v1[1], v1[2], v1[3], pos[pct])
                    if overlap < 0.5:
                        continue
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            v = [symbol.bounding_box.vertices[i] for i in range(4)]
                            v = np.array([[v[0].x, v[0].y],
                                          [v[1].x, v[1].y],
                                          [v[2].x, v[2].y],
                                          [v[3].x, v[3].y]])
                            v = v - np.array([pos[pct]['canvas_w_offset'], pos[pct]['canvas_h_offset']])
                            v = v * 100 / pct
                            width = 0.5 * (v[1, 0] - v[0, 0] + v[2, 0] - v[3, 0])
                            height = 0.5 * (v[3, 1] - v[0, 1] + v[2, 1] - v[1, 1])
                            center = v.mean(axis=0).astype(int)

                            d = {'v': v, 't': symbol.text, 's': symbol.confidence, 'width': int(width),
                                 'height': int(height),
                                 'center': center}
                            if symbol.property.detected_break.type == 1:
                                d['space'] = True
                            symbols[pct].append(d)
        return symbols


def find_top_2_symbols_by_len(symbols):
    lens = [len(i) for i in symbols]
    c = collections.Counter(lens)
    if len(c) == 1:
        return symbols, [], []

    l1 = c.most_common(1)[0][0]
    l2 = c.most_common(2)[1][0]

    leftovers = []
    frequent1 = []
    frequent2 = []
    for s in symbols:
        if len(s) == l1:
            frequent1.append(s)
        elif len(s) == l2:
            frequent2.append(s)
        else:
            leftovers.append(s)

    return frequent1, frequent2, leftovers


def find_common_str(symbols):
    lens = [len(s) for s in symbols]
    assert (len(lens) > 1)
    assert (len(collections.Counter(lens)) == 1)

    common_str = []
    char_scores = []
    count_score = 1.0
    for i in range(lens[0]):  # pick a position
        chars = [s[i] for s in symbols]  # pick all characters for that position
        centers = np.array([c['center'] for c in chars])
        width = np.array([c['width'] for c in chars])

        # chars = [f"{s[i]['s']:.2f}"  for s in symbols]
        center_median = np.median(centers, axis=0)
        #center_std = np.std(centers, axis=0)

        width_median = np.median(width, axis=0)
        #width_std = np.std(width, axis=0)

        # print(center_median, center_std, width_median, width_std)
        chars = [c for c in chars if abs(c['center'][0] - center_median[0]) < width_median / 5]
        # char_scores.append(min([c['s'] for c in chars]))
        char_vals = [c['t'] for c in chars]
        # count_score = min(count_score, len(chars)/24)

        if len(chars) > 1:
            c = collections.Counter(char_vals)
            c = c.most_common(1)[0][0]
            common_str.append(c)
            char_c = [k['s'] for k in chars if k['t'] == c]
            char_scores.append(min(char_c))
            count_score = min(count_score, len(char_c) / 24)

    if '@' not in common_str or '.' not in common_str or len({common_str[0], common_str[-1]}.intersection({'@', '.'})) != 0:
        count_score = 0

    if len({':', '#', '!', '%', '^', '&', '(', ')', '{', '}', '[', ']', '\'', ':', ';', '?', '<', '>'}.intersection(
            set(common_str))) != 0:
        count_score = 0

    return ''.join(common_str), count_score, char_scores

