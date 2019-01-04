#!/usr/bin/env python3
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import time
from pathlib import Path

import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from faster_rcnn import FasterRCNN
from faster_rcnn import Config
from faster_rcnn import iou
from imgs_to_roi_features import (
    format_img_channels,
    format_img_size,
    imgs_to_roi_features,
)
from create_retrieval_db import best_bbox


def get_map(pred, gt):
    T = {}
    P = {}

    for bbox in gt:
        bbox["bbox_matched"] = False

    pred_probs = np.array([s["prob"] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box["class"]
        pred_x1 = pred_box["x1"]
        pred_x2 = pred_box["x2"]
        pred_y1 = pred_box["y1"]
        pred_y2 = pred_box["y2"]
        pred_prob = pred_box["prob"]
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box["class"]
            gt_x1 = gt_box["x1"]
            gt_x2 = gt_box["x2"]
            gt_y1 = gt_box["y1"]
            gt_y2 = gt_box["y2"]
            gt_seen = gt_box["bbox_matched"]
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou_map = iou(
                (pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2)
            )
            if iou_map >= 0.5:
                found_match = True
                gt_box["bbox_matched"] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box["bbox_matched"]:  # and not gt_box['difficult']:
            if gt_box["class"] not in P:
                P[gt_box["class"]] = []
                T[gt_box["class"]] = []
            print(f'Some gt box has not been associated to {gt_box["path"]}')
            T[gt_box["class"]].append(1)
            P[gt_box["class"]].append(0)
    return T, P


def format_img_map(img, C):
    """Format image for mAP. Resize original image to C.im_size (300 in here)

    Args:
      img: cv2 image
      C: config

    Returns:
      img: Scaled and normalized image with expanding dimension
      fx: ratio for width scaling
      fy: ratio for height scaling
    """
    img, ratio, fx, fy = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, fx, fy


def data_to_dict(l):
    l = l.strip().split(",")
    return {
        "path": l[0],
        "x1": int(l[1]),
        "y1": int(l[2]),
        "x2": int(l[3]),
        "y2": int(l[4]),
        "class": l[5],
    }


if __name__ == "__main__":

    config_output_filename = "data/instre_monuments/model_vgg_config.pickle"

    with open(config_output_filename, "rb") as f_in:
        C = pickle.load(f_in)

    test_path = (
        "data/instre_monuments/annotations_test.txt"
    )  # Test data (annotation file)

    with open(test_path) as f:
        test_imgs = map(data_to_dict, f.readlines())

    T = {}
    P = {}
    mAPs = []

    imgs_paths = list(map(lambda img_data: img_data["path"], test_imgs))
    with tqdm(total=len(imgs_paths)) as pbar:
        feats = imgs_to_roi_features(imgs_paths, C, 0.7, on_each_iter=pbar.update)

    for idx, img_data in enumerate(test_imgs):
        # img_data = (path, (x1,y1,x2,y2), class)

        t, p = {}, {}

        result = None
        if img_data["path"] in feats:
            result = feats[img_data["path"]]

            jk = best_bbox(result)

            x1, y1, x2, y2 = result[0][jk]
            prob = result[1][jk][0]
            key = result[1][jk][1]

            det = {"x1": x1, "x2": x2, "y1": y1, "y2": y2, "class": key, "prob": prob}
            t, p = get_map([det], [img_data])

        else:
            t, p = get_map([], [img_data])

        for key in t.keys():
            if key not in T:
                T[key] = []
                P[key] = []
            T[key].extend(t[key])
            P[key].extend(p[key])
        all_aps = []
        for key in T.keys():
            ap = average_precision_score(T[key], P[key])
            print("{} AP: {}".format(key, ap))
            all_aps.append(ap)
        print("mAP = {}".format(np.mean(np.array(all_aps))))
        mAPs.append(np.mean(np.array(all_aps)))
        # print(T)
        # print(P)

    print()
    print("mean average precision:", np.mean(np.array(mAPs)))

    mAP = [mAP for mAP in mAPs if str(mAP) != "nan"]
    mean_average_prec = round(np.mean(np.array(mAP)), 3)
    print(f"The mean average precision is {mean_average_prec}")
