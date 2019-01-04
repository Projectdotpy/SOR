import cv2

import numpy as np

from keras import backend as K
from keras.layers import Input
from keras.models import Model

from faster_rcnn import FasterRCNN

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


def format_img_channels(img, C):
    """ formats the image channels based on config """
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
    return img


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
    fx = width / float(new_width)
    fy = height / float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio, fx, fy


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio, fx, fy = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


def imgs_to_roi_features(imgs_paths, C, bbox_threshold, on_each_iter=None, train=False):
    """Given a set of images paths transforms them to the
    RoI pooled feature of the most confident object in the image
    
    Arguments:
        imgs_paths {list(file_paths)} -- List of the file paths the imgs are found
        C {Config} -- Configuration object taken from pickle
    
    Returns:
        {
            '<img_path>': ( list((x1, y1, x2, y2)), list((prob, class)), list(feature (7x7x512)) )
        }
    """

    if not train:
        # turn off any data augmentation
        C.use_horizontal_flips = False
        C.use_vertical_flips = False
        C.rot_90 = False

    model_frcnn = FasterRCNN()
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

    classifier = model_frcnn.classifier_layer(
        feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping)
    )

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    feature_extraction_input = Input(shape=(1, 4))
    roi_pooling = model_frcnn.roi_pooling_layer(
        feature_map_input, feature_extraction_input, 1, nb_classes=len(C.class_mapping)
    )
    model_roi_pooling = Model(
        [feature_map_input, feature_extraction_input], roi_pooling
    )

    try:
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)
    except Exception:
        # When calling this function from the server, given that
        # it is multithreaded, an exception is raised since the model's
        # weights were already loaded.
        # A better approach would be to create the model only once
        pass

    model_rpn.compile(optimizer="sgd", loss="mse")
    model_classifier.compile(optimizer="sgd", loss="mse")

    # Switch key value for class mapping
    class_mapping = C.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}

    features_per_class = {}
    metadata_per_class = {}

    result = {}
    for img_path in imgs_paths:
        img = cv2.imread(img_path)
        X, ratio = format_img(img, C)

        X = np.transpose(X, (0, 2, 3, 1))

        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        [Y1, Y2, F] = model_rpn.predict(X)

        # Get bboxes by applying NMS
        # R.shape = (300, 4)
        R = model_frcnn.rpn_to_roi(
            Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7
        )

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        feature_img_box_mapping = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk : C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, : curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1] :, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):
                # Ignore 'bg' class
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(
                    P_cls[0, ii, :]
                ) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num : 4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = model_frcnn.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                feature_img_box_mapping[
                    (
                        C.rpn_stride * x,
                        C.rpn_stride * y,
                        C.rpn_stride * (x + w),
                        C.rpn_stride * (y + h),
                    )
                ] = ROIs[0, ii, :]
                bboxes[cls_name].append(
                    [
                        C.rpn_stride * x,
                        C.rpn_stride * y,
                        C.rpn_stride * (x + w),
                        C.rpn_stride * (y + h),
                    ]
                )
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = model_frcnn.non_max_suppression_fast(
                bbox, np.array(probs[key]), overlap_thresh=0.2
            )
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                # Calculate real coordinates on original image
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(
                    ratio, x1, y1, x2, y2
                )

                features = model_roi_pooling.predict(
                    [
                        F,
                        np.reshape(
                            feature_img_box_mapping[(x1, y1, x2, y2)], (1, 1, 4)
                        ),
                    ]
                )
                features = features.reshape((-1,))

                result[img_path] = result.get(img_path, ([], [], []))
                result[img_path][0].append((real_x1, real_y1, real_x2, real_y2))
                result[img_path][1].append((new_probs[jk], key))
                result[img_path][2].append(features)

        if on_each_iter:
            on_each_iter()
    return result
