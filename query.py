#!/usr/bin/env python3

import pickle

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from faster_rcnn import Config
from imgs_to_roi_features import imgs_to_roi_features
from create_retrieval_db import best_bbox


def images_similar_to(q_img_path, features_per_class, metadata_per_class, C):
    result = imgs_to_roi_features([q_img_path], C, bbox_threshold=0.7)
    instance = result[q_img_path]
    best_i = best_bbox(instance)

    best_feature = instance[2][best_i]
    claz = instance[1][best_i][1]

    pool = features_per_class[claz]

    sims = cosine_similarity(pool, np.array([best_feature])).reshape(-1)
    top = np.argsort(sims)[::-1]

    similar_images = [metadata_per_class[claz][im][0] for im in top]

    return similar_images, result
