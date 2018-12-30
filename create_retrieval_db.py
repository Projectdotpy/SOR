#!/usr/bin/env python3

import pickle

import numpy as np

from glob import glob
from pathlib import Path

from tqdm import tqdm

from faster_rcnn import Config
from imgs_to_roi_features import imgs_to_roi_features


def best_bbox(instance, n=1):
    """Returns the index of the box having the highest confidence if n=1
    Otherwise a list of the best indices
    """
    result = list(
        map(
            lambda t: t[0],
            sorted(enumerate(instance[1]), key=lambda t: t[1][0], reverse=True),
        )
    )
    if not n:
        return result
    elif n == 1:
        return result[0]
    return result[:n]


if __name__ == "__main__":
    dataset_imgs = list(glob("data/instre_monuments/train/*.jpg"))

    with open("data/instre_monuments/model_vgg_config.pickle", "rb") as f_in:
        C = pickle.load(f_in)

    with tqdm(total=len(dataset_imgs)) as pbar:
        result = imgs_to_roi_features(
            dataset_imgs, C, bbox_threshold=0.7, on_each_iter=pbar.update
        )

    with open("imgs_to_roi", "wb") as f:
        pickle.dump(result, f)

    """
    features_per_class = {
        '<class_name>': ndarray (n_k x 25088)
    }
    
    metadata_per_class = {
        '<class_name>': list((img_name, bbox))
    }
    """
    features_per_class = {}
    metadata_per_class = {}

    for img in result:
        best_i = best_bbox(result[img])
        claz = result[img][1][best_i][1]

        metadata_per_class[claz] = metadata_per_class.get(claz, [])
        features_per_class[claz] = features_per_class.get(claz, [])

        metadata_per_class[claz].append((img, result[img][0][best_i]))
        features_per_class[claz].append(result[img][2][best_i])

    for claz in features_per_class:
        features_per_class[claz] = np.array(features_per_class[claz])

    retrieval_db_path = Path("retrieval_db")

    with open(retrieval_db_path / "features_per_class", "wb") as f:
        pickle.dump(features_per_class, f)

    with open(retrieval_db_path / "metadata_per_class", "wb") as f:
        pickle.dump(metadata_per_class, f)
