#!/usr/bin/env python3

import glob, os, shutil, random

from pathlib import Path

DATA_PATH = Path('data/instre_monuments')
TEST_PATH = DATA_PATH / 'test'
TRAIN_PATH = DATA_PATH / 'train'

os.mkdir(TEST_PATH)
os.mkdir(TRAIN_PATH)

TRAIN_PERC = 0.8

annotations = list(glob.glob(str(DATA_PATH / '*.txt')))
random.Random(42).shuffle(annotations)

train_filenames = annotations[:int(TRAIN_PERC * len(annotations))]
test_filenames = annotations[int(TRAIN_PERC * len(annotations)):]

for fname in train_filenames:
    fname = Path(fname)
    os.rename(fname, TRAIN_PATH / fname.name) # mv txt
    fname = fname.with_suffix('.jpg')
    os.rename(fname, TRAIN_PATH / fname.name)


for fname in test_filenames:
    fname = Path(fname)
    os.rename(fname, TEST_PATH / fname.name) # mv txt
    fname = fname.with_suffix('.jpg')
    os.rename(fname, TEST_PATH / fname.name)

dirs = os.listdir(DATA_PATH)
dirs.remove('train')
dirs.remove('test')

for d in dirs:
    shutil.rmtree(d)

train_filenames = list(glob.glob(str(TRAIN_PATH / '*.txt')))
test_filenames = list(glob.glob(str(TEST_PATH / '*.txt')))

annotations_train_f = DATA_PATH / 'annotations_train.txt'

annotations_test_f = DATA_PATH / 'annotations_test.txt'

def f_to_string(fname):
    ann = ''
    with open(fname) as f:
        ann = f.readlines()[0].strip()
    x,y,w,h = ann.split()
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + int(w)
    y2 = y1 + int(h)

    class_name = Path(fname).name
    class_name = class_name[:class_name.rfind('_')]
    return ','.join([
                    str(Path(fname).with_suffix('.jpg')),
                    str(x1),
                    str(y1),
                    str(x2),
                    str(y2),
                    class_name])


with open(annotations_train_f, 'w') as f:
    for fname in train_filenames:
        f.write(f_to_string(fname) + '\n')

with open(annotations_test_f, 'w') as f:
    for fname in test_filenames:
        f.write(f_to_string(fname) + '\n')
