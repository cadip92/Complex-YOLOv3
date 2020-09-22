import numpy as np
import argparse
import cv2
import time, json, os, math, sys
from pprint import pprint
from easydict import EasyDict as edict

import utils.utils as utils
from models import *
import torch.utils.data as torch_data

import utils.kitti_utils as kitti_utils
import utils.kitti_aug_utils as aug_utils
import utils.kitti_bev_utils as bev_utils
from utils.kitti_yolo_dataset import KittiYOLODataset
import utils.config as cnf
import utils.mayavi_viewer as mview

def predictions_to_kitti_format(img_detections, calib, img_shape_2d, img_size, RGB_Map=None):
    predictions = np.zeros([50, 7], dtype=np.float32)
    count = 0
    for detections in img_detections:
        if detections is None:
            continue
        # Rescale boxes to original image
        for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
            yaw = np.arctan2(im, re)
            predictions[count, :] = cls_pred, x/img_size, y/img_size, w/img_size, l/img_size, im, re
            count += 1

    predictions = bev_utils.inverse_yolo_target(predictions, cnf.boundary)
    if predictions.shape[0]:
        predictions[:, 1:] = aug_utils.lidar_to_camera_box(predictions[:, 1:], calib.V2C, calib.R0, calib.P)

    objects_new = []
    corners3d = []
    for index, l in enumerate(predictions):

        str = "Pedestrian"
        if l[0] == 0:str="Car"
        elif l[0] == 1:str="Pedestrian"
        elif l[0] == 2: str="Cyclist"
        else:str = "DontCare"
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = kitti_utils.Object3d(line)
        obj.t = l[1:4]
        obj.h,obj.w,obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))
    
        _, corners_3d = kitti_utils.compute_box_3d(obj, calib.P)
        corners3d.append(corners_3d)
        objects_new.append(obj)

    if len(corners3d) > 0:
        corners3d = np.array(corners3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape_2d[1] * 0.8, img_boxes_h < img_shape_2d[0] * 0.8)

    for i, obj in enumerate(objects_new):
        x, z, ry = obj.t[0], obj.t[2], obj.ry
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        obj.alpha = alpha
        obj.box2d = img_boxes[i, :]

    if RGB_Map is not None:
        labels, noObjectLabels = kitti_utils.read_labels_for_bevbox(objects_new)    
        if not noObjectLabels:
            labels[:, 1:] = aug_utils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P) # convert rect cam to velo cord

        target = bev_utils.build_yolo_target(labels)
        utils.draw_box_in_bev(RGB_Map, target)

    return objects_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default=None, type=str, help='Configuration file')
    parser = parser.parse_args()

    try:
        if parser.config is not None:
            with open(parser.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(parser.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    FLAGS = edict(config_args_dict)

    if FLAGS.checkpoint_file == "":
        print("ERROR: Checkpoint file is not specified", file=sys.stderr)
        exit(1)

    pprint(FLAGS)

    classes = utils.load_classes("data/classes.names")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(FLAGS.model_def, img_size=cnf.BEV_WIDTH).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(FLAGS.checkpoint_file))
    # Eval mode
    model.eval()
    
    dataset = KittiYOLODataset(cnf.root_dir, split=FLAGS.split, mode='TEST', folder="sampledata", data_aug=False)
    data_loader = torch_data.DataLoader(dataset, 1, shuffle=False)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    start_time = time.time()                        
    for index, (img_paths, bev_maps) in enumerate(data_loader):
        
        # Configure bev image
        input_imgs = Variable(bev_maps.type(Tensor))

        # Get detections 
        with torch.no_grad():
            detections = model(input_imgs)
            detections = utils.non_max_suppression_rotated_bbox(detections, FLAGS.conf_thres, FLAGS.nms_thres) 
        
        end_time = time.time()
        print(f"FPS: {(1.0/(end_time-start_time)):0.2f}")
        start_time = end_time

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        bev_maps = torch.squeeze(bev_maps).numpy()

        RGB_Map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
        RGB_Map[:, :, 2] = bev_maps[0, :, :]  # r_map
        RGB_Map[:, :, 1] = bev_maps[1, :, :]  # g_map
        RGB_Map[:, :, 0] = bev_maps[2, :, :]  # b_map
        
        RGB_Map *= 255
        RGB_Map = RGB_Map.astype(np.uint8)
        
        for detections in img_detections:
            if detections is None:
                continue

            # Rescale boxes to original image
            detections = utils.rescale_boxes(detections, cnf.BEV_WIDTH, RGB_Map.shape[:2])
            for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                bev_utils.drawRotatedBox(RGB_Map, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

        img2d = cv2.imread(img_paths[0])
        calib = kitti_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
        objects_pred = predictions_to_kitti_format(img_detections, calib, img2d.shape, cnf.BEV_WIDTH)  
        
        img2d = mview.show_image_with_boxes(img2d, objects_pred, calib, False)

        cv2.imwrite("test_%s.jpeg" % index, img2d)

        if cv2.waitKey(0) & 0xFF == 27:
            break