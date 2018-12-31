import pickle

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from faster_rcnn import Config
from imgs_to_roi_features import imgs_to_roi_features
from create_retrieval_db import best_bbox


def images_similar_to(q_img_path, features_per_class, metadata_per_class, C):
    result = imgs_to_roi_features([q_img_path], C, bbox_threshold=0.7)
    if not q_img_path in result:
        return [], result
    instance = result[q_img_path]
    best_is = best_bbox(instance, n=None)

    similar_images = []
    for best_i in best_is:
        best_feature = instance[2][best_i]
        claz = instance[1][best_i][1]

        pool = features_per_class[claz]

        sims = cosine_similarity(pool, np.array([best_feature])).reshape(-1)
        top = np.argsort(sims)[::-1]
        similar_images_in_claz = [(metadata_per_class[claz][im][0], sims[im]) for im in top]

        similar_images += similar_images_in_claz

    similar_images = map(lambda t: t[0], sorted(similar_images, key=lambda t: t[1], reverse=True))

    return similar_images, result
