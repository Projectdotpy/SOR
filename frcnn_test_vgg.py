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


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


def concat_query_result2(image_names):
    images = []
    max_width = 0  # find the max width of all the images
    max_height = 0

    for name in image_names:
        # open all images and find their sizes
        images.append(cv2.imread(name))
        max_width = max(max_width, images[-1].shape[1])
        max_height = max(max_height, images[-1].shape[0])

    last_h_pixel = int(max_width * len(image_names) / 2 + 1)
    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((max_height * 2, last_h_pixel, 3), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    current_x = 0
    for image in images:
        if current_x + image.shape[1] > last_h_pixel:
            current_y += max_height
            current_x = 0

        final_image[current_y:image.shape[0] + current_y, current_x:current_x + image.shape[1], :] = image
        current_x += image.shape[1]

    return final_image


def concat_query_result(image_names):
    images = []
    max_width = 0  # find the max width of all the images
    total_height = 0  # the total height of the images (vertical stacking)

    for name in image_names:
        # open all images and find their sizes
        images.append(cv2.imread(name))
        if images[-1].shape[1] > max_width:
            max_width = images[-1].shape[1]
        total_height += images[-1].shape[0]

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in images:
        # add an image to the final array and increment the y coordinate
        final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
        current_y += image.shape[0]

    return final_image


def get_map(pred, gt, f):
    T = {}
    P = {}
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1'] / fx
            gt_x2 = gt_box['x2'] / fx
            gt_y1 = gt_box['y1'] / fy
            gt_y2 = gt_box['y2'] / fy
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou_map = iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            if iou_map >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box['bbox_matched']:  # and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)
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

    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        f = img_min_side / width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side / height
        new_width = int(f * width)
        new_height = int(img_min_side)
    fx = width / float(new_width)
    fy = height / float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # Change image channel from BGR to RGB
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    # Change img shape from (height, width, channel) to (channel, height, width)
    img = np.transpose(img, (2, 0, 1))
    # Expand one dimension at axis 0
    # img shape becames (1, channel, height, width)
    img = np.expand_dims(img, axis=0)
    return img, fx, fy


if __name__ == '__main__':

    model_frcnn = FasterRCNN()

    base_path = 'data/instre_monuments/'

    test_path = 'data/instre_monuments/annotations_test.txt'  # Test data (annotation file)

    test_base_path = 'data/instre_monuments/test'  # Directory to save the test images

    train_base_path = 'data/instre_monuments/train'

    config_output_filename = os.path.join(base_path, 'model_vgg_config.pickle')

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    # # Test

    C.num_rois = 1
    num_features = 512

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = model_frcnn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = model_frcnn.rpn_layer(shared_layers, num_anchors)

    classifier = model_frcnn.classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    roi_pooling = model_frcnn.roi_pooling_layer(feature_map_input, roi_input, C.num_rois,
                                                nb_classes=len(C.class_mapping))
    model_roi_pooling = Model([feature_map_input, roi_input], roi_pooling)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    # Switch key value for class mapping
    class_mapping = C.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    train_imgs = list(filter(lambda e: Path(e).suffix == '.jpg', os.listdir(train_base_path)))
    test_imgs = os.listdir(test_base_path)

    imgs_path = []
    for i in range(12):
        idx = np.random.randint(len(test_imgs))
        imgs_path.append(test_imgs[idx])

    all_imgs = []

    classes = {}

    # If the box classification value is less than this, we ignore this box
    bbox_threshold = 0.7

    retrieval_db_path = Path('retrieval_db')

    query = True

    '''
    features_per_class = {
        '<class_name>': Matrix (n_k x 25088)
    }
    
    metadata_per_class = {
        '<class_name>': list((img_name, bbox))
    }
    '''

    features_per_class = {}
    metadata_per_class = {}

    random.Random(42).shuffle(train_imgs)

    if query:
        with open(retrieval_db_path / 'features_per_class', 'rb') as f:
            features_per_class = pickle.load(f)

        with open(retrieval_db_path / 'metadata_per_class', 'rb') as f:
            metadata_per_class = pickle.load(f)

        train_imgs = list(filter(lambda e: Path(e).suffix == '.jpg', os.listdir('query')))
        train_base_path = 'query/'

    # TODO: Remove this
    max_imgs = 5
    train_imgs = train_imgs[:max_imgs]
    # END Remove this

    for img_name in tqdm(train_imgs):

        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        st = time.time()

        filepath = os.path.join(train_base_path, img_name)

        img = cv2.imread(filepath)

        X, ratio = format_img(img, C)

        X = np.transpose(X, (0, 2, 3, 1))

        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        [Y1, Y2, F] = model_rpn.predict(X)

        # Get bboxes by applying NMS
        # R.shape = (300, 4)
        R = model_frcnn.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        feature_img_box_mapping = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):
                # Ignore 'bg' class
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = model_frcnn.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                feature_img_box_mapping[
                    (C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h))] = ROIs[0, ii,
                                                                                                            :]
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        best_prob_for_class = {cl: max(probs[cl]) for cl in probs}

        if not best_prob_for_class:
            print(f'No class found for image {img_name}')
            continue

        key = max(best_prob_for_class, key=best_prob_for_class.get)

        bbox = np.array(bboxes[key])
        new_boxes, new_probs = model_frcnn.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(1):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            features = model_roi_pooling.predict([F, np.reshape(feature_img_box_mapping[(x1, y1, x2, y2)], (1, 1, 4))])
            features = features.reshape((-1,))

            if not query:
                # Append feature of current box
                features_per_class[key] = features_per_class.get(key, [])
                features_per_class[key].append(features)

            # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            if not query:
                # Append coordinates and filename
                metadata_per_class[key] = metadata_per_class.get(key, [])
                metadata_per_class[key].append((img_name, (real_x1, real_y1, real_x2, real_y2)))

            cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                          (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 4)

            textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
            all_dets.append((key, 100 * new_probs[jk]))

            (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            textOrg = (real_x1, real_y1 - 0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 1)
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        if query:
            pool = features_per_class[key]
            sims = cosine_similarity(pool, np.array([features])).reshape(-1)
            top = np.argsort(sims)[::-1]

            tops = [str(Path('data/instre_monuments/train') / metadata_per_class[key][im][0]) for im in top[:10]]
            print(tops)
            concat_results = concat_query_result2(tops)

            print('Elapsed time = {}'.format(time.time() - st))
            print(all_dets)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show(block=False)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(concat_results, cv2.COLOR_BGR2RGB))
            plt.show()
            exit()

    # List to numpy n-dimensional array
    for key in features_per_class:
        features_per_class[key] = np.array(features_per_class[key])

    os.mkdir(retrieval_db_path)

    with open(retrieval_db_path / 'features_per_class', 'wb') as f:
        pickle.dump(features_per_class, f)

    with open(retrieval_db_path / 'metadata_per_class', 'wb') as f:
        pickle.dump(metadata_per_class, f)
    print("done")
    exit()
    # #### Measure mAP

    print(class_mapping)

    # This might takes a while to parser the data
    test_imgs, _, _ = model_frcnn.get_data(test_path)

    T = {}
    P = {}
    mAPs = []
    for idx, img_data in enumerate(test_imgs):
        print('{}/{}'.format(idx, len(test_imgs)))
        st = time.time()
        filepath = img_data['filepath']

        img = cv2.imread(filepath)

        X, fx, fy = format_img_map(img, C)

        # Change X (img) shape from (1, channel, height, width) to (1, height, width, channel)
        X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = model_frcnn.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            # Calculate all classes' bboxes coordinates on resized image (300, 400)
            # Drop 'bg' classes bboxes
            for ii in range(P_cls.shape[1]):

                # If class name is 'bg', continue
                if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                # Get class name
                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            # Apply non-max-suppression on final bboxes to get the output bounding boxe
            new_boxes, new_probs = model_frcnn.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
                all_dets.append(det)

        print('Elapsed time = {}'.format(time.time() - st))
        t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))
        for key in t.keys():
            if key not in T:
                T[key] = []
                P[key] = []
            T[key].extend(t[key])
            P[key].extend(p[key])
        all_aps = []
        for key in T.keys():
            ap = average_precision_score(T[key], P[key])
            print('{} AP: {}'.format(key, ap))
            all_aps.append(ap)
        print('mAP = {}'.format(np.mean(np.array(all_aps))))
        mAPs.append(np.mean(np.array(all_aps)))
        # print(T)
        # print(P)

    print()
    print('mean average precision:', np.mean(np.array(mAPs)))

    mAP = [mAP for mAP in mAPs if str(mAP) != 'nan']
    mean_average_prec = round(np.mean(np.array(mAP)), 3)
    print('After training %dk batches, the mean average precision is %0.3f' % (len(record_df), mean_average_prec))
