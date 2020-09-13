from __future__ import division

from models import *
from utils.utils import *

import os, sys, time, datetime, argparse, json
from pprint import pprint
from easydict import EasyDict as edict
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

import utils.config as cnf
from utils.kitti_yolo_dataset import KittiYOLODataset

def evaluate(model, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    split='valid'
    dataset = KittiYOLODataset(cnf.root_dir, split=split, mode='EVAL', folder='training', data_aug=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    pred_time, det_time = [], []
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            start_time = time.time()
            outputs = model(imgs)
            outputs = non_max_suppression_rotated_bbox(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            end_pred_time = time.time()

        sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=iou_thres)
        end_time = time.time()
        pred_time.append(int(round((end_pred_time - start_time) * 1000)) / batch_size)
        det_time.append(int(round((end_time - start_time) * 1000)) / batch_size)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class, np.mean(pred_time), np.mean(det_time)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_classes("data/classes.names")

    # Initiate model
    model = Darknet(FLAGS.model_def).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(FLAGS.checkpoint_file))

    print("Compute mAP...")
    precision, recall, AP, f1, ap_class, avg_pred_time, avg_det_time = evaluate(model, iou_thres=FLAGS.iou_thres,
                                                                                conf_thres=FLAGS.conf_thres,
                                                                                nms_thres=FLAGS.nms_thres,
                                                                                img_size=cnf.BEV_WIDTH,
                                                                                batch_size=FLAGS.batch_size)

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

    print("Average time for prediction and NMS is %.2f ms" % avg_pred_time)
    print("Average total time for detection is %.2f ms" % avg_det_time)