#!/usr/bin/env python3
import pickle
from faster_rcnn import Config

with open('data/instre_monuments/model_vgg_config.pickle', 'rb') as f:
    C = pickle.load(f)
C.model_path = 'model/model_frcnn_vgg.hdf5'
with open('data/instre_monuments/model_vgg_config.pickle', 'wb') as f:
    pickle.dump(C, f)