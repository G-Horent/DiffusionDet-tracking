import os
import pandas as pd

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import random
import cv2

MOT_COLUMNS_NAMES = ['frame_nb', 'id_nb', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf_score', 'class',
                     'visibility']

MOT_CLASS_NAMES = ['other','pedestrian', 'person on vehicle', 'car', 'bicycle', 'Motorbike', 'NMVehicle', 'Static person',
                   'Distractor', 'occluder', 'occluder_ground', 'occluder_full', 'reflection']


def get_MOT17_train():
    path_dir = 'datasets/MOT17/train'
    dataset_dics = []
    list_dir = os.listdir(path_dir)
    list_dir = [x for x in list_dir if x.endswith('FRCNN')]
    for directory in list_dir:
        file_gt = pd.read_csv(os.path.join(path_dir, directory, 'gt', 'gt.txt'), sep=',', header=None,
                              names=MOT_COLUMNS_NAMES)

        imgs_names = os.listdir(os.path.join(path_dir, directory, 'img1'))

        if not directory.startswith('MOT17-05'):
            width_im, height_im = 1920, 1080
        else:
            width_im, height_im = 640, 480

        for img_name in imgs_names:
            record = {}
            record["file_name"] = os.path.join(path_dir, directory, 'img1', img_name)
            record["image_id"] = directory + '/' + img_name
            record["height"] = height_im
            record["width"] = width_im
            

            frame_nb = int(img_name.split('.')[0])
            detections_in_img = file_gt[file_gt["frame_nb"] == frame_nb]
            annotations = []

            for idx, row in detections_in_img.iterrows():
                curr_detection = {}
                curr_detection["bbox"] = [row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height']]
                curr_detection["bbox_mode"] = 1
                curr_detection["category_id"] = int(row['class'])
                curr_detection['ped_id'] = row['id_nb']

                annotations.append(curr_detection)

            record['annotations'] = annotations

            dataset_dics.append(record)

    return dataset_dics


def get_MOT17_test():
    path_dir = 'datasets/MOT17/test'
    dataset_dics = []
    list_dir = os.listdir(path_dir)
    list_dir = [x for x in list_dir if x.endswith('FRCNN')]
    for directory in list_dir:
        file_gt = pd.read_csv(os.path.join(path_dir, directory, 'gt', 'gt.txt'), sep=',', header=None,
                              names=MOT_COLUMNS_NAMES)

        imgs_names = os.listdir(os.path.join(path_dir, directory, 'img1'))

        if not directory.startswith('MOT17-06'):
            width_im, height_im = 1920, 1080
        else:
            width_im, height_im = 640, 480

        for img_name in imgs_names:
            record = {}
            record["file_name"] = os.path.join(path_dir, directory, 'img1', img_name)
            record["img_id"] = directory + '/' + img_name
            record["height"] = height_im
            record["width"] = width_im

            frame_nb = int(img_name.split('.')[0])
            detections_in_img = file_gt[file_gt["frame_nb"] == frame_nb]
            annotations = []

            for idx, row in detections_in_img.iterrows():
                curr_detection = {}
                curr_detection["bbox"] = [row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height']]
                curr_detection["bbox_mode"] = 1
                curr_detection["category_id"] = int(row['class']) - 1

                annotations.append(curr_detection)

            record['annotations'] = annotations

            dataset_dics.append(record)

    return dataset_dics