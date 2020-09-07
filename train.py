from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.kitti_yolo_dataset import KittiYOLODataset
from eval_mAP import evaluate
from easydict import EasyDict as edict
from pprint import pprint

from terminaltables import AsciiTable
import os, sys, time, datetime, argparse, json

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

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

    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    class_names = load_classes("data/classes.names")

    # Initiate model
    model = Darknet(FLAGS.model_def, img_size=cnf.BEV_WIDTH).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if FLAGS.pretrained_weights != "":
        if FLAGS.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(FLAGS.pretrained_weights))
        else:
            model.load_darknet_weights(FLAGS.pretrained_weights)

    # No.of parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters : ", pytorch_total_params, " or ~%.2fM parameters" % (pytorch_total_params/1000000.0))

    # Get dataloader
    dataset = KittiYOLODataset(
        cnf.root_dir,
        split='train',
        mode='TRAIN',
        folder='training',
        data_aug=True,
        multiscale=FLAGS.multiscale_training
    )

    dataloader = DataLoader(
        dataset,
        FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = optim.Adam(model.parameters(),
                           lr=FLAGS.learning_rate)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "im",
        "re",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    max_mAP = 0.0
    for epoch in range(0, FLAGS.epochs, 1):
        model.train()
        start_time = time.time()

        # Print batch progress as a Progress Bar
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Training for epoch %d started" %epoch)):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % FLAGS.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, FLAGS.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # Logging the table after the final batch for each epoch
        log_str += AsciiTable(metric_table).table
        log_str += f"\nTotal loss {loss.item()}"

        # Determine approximate time left for epoch
        epoch_batches_left = len(dataloader) - (batch_i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        log_str += f"\n---- ETA {time_left}"

        print(log_str)

        # Performing evaluation after every 2 Epochs

        if epoch % FLAGS.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                iou_thres=FLAGS.iou_thres,
                conf_thres=FLAGS.conf_thres,
                nms_thres=FLAGS.nms_thres,
                img_size=cnf.BEV_WIDTH,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            #if epoch % FLAGS.checkpoint_interval == 0:
            if AP.mean() > max_mAP:
                print("Saving model. Epoch No. %d" % epoch)
                torch.save(model.state_dict(), FLAGS.checkpoint_file)
                max_mAP = AP.mean()
