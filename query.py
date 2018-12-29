#!/usr/bin/env python3

import pickle

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from faster_rcnn import Config
from imgs_to_roi_features import imgs_to_roi_features
from create_retrieval_db import best_bbox


def images_similar_to(q_img_path, features_per_class, metadata_per_class):
    result = imgs_to_roi_features([q_img_path], C, bbox_threshold=0.7)
    instance = result[q_img_path]
    best_i = best_bbox(instance)

    best_feature = instance[2][best_i]
    claz = instance[1][best_i][1]

    pool = features_per_class[claz]

    sims = cosine_similarity(pool, np.array([best_feature])).reshape(-1)
    top = np.argsort(sims)[::-1]

    similar_images = [metadata_per_class[claz][im][0] for im in top]

    return similar_images


if __name__ == "__main__":
    with open("data/instre_monuments/model_vgg_config.pickle", "rb") as f_in:
        C = pickle.load(f_in)

    with open("retrieval_db/features_per_class", "rb") as f:
        features_per_class = pickle.load(f)

    with open("retrieval_db/metadata_per_class", "rb") as f:
        metadata_per_class = pickle.load(f)

    print(
        images_similar_to(
            "query/colosseum_038.jpg", features_per_class, metadata_per_class
        )
    )
