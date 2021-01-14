Downloading the dataset :

1) Create a directory 'data/KITTI'

2) Visit the official KITTI dataset on http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

3) Download the files Camera calibration matrices of object data set (16 MB), Training labels of object data set (5 MB), Velodyne point clouds (29 GB) and Left color images of object data set (12 GB).

4) Re-arrange the dowloaded files in the following order :

	└── data/KITTI/object
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne           
		   
5) Create a new directory 'data/KITTI/Imagesets' and create three files test.txt(specify the filenames which should be considered for testing. For e.g. 000000, 000001), train.txt and valid.txt.

6) Create the file 'classes.names' in 'data/KITTI' directory and specify 'Car', 'Pedestrian' and 'Cyclist', each on a new line.

Steps for Training :

1) Modify the "model_def" and "checkpoint_file" in the config.json file before execution
for e.g. "config/complex_yolov3.cfg" is the model configuration file for Complex YOLO v3. The config files are available in the 'config' directory. The checkpoints for previously trained models are available in the 'checkpoints' file.

2) Type the command "python train.py --config config.json" to start training

Step for Evaluation :

1) Modify the "model_def" and "checkpoint_file" in the config.json

2) Type the command "python eval_mAP.py --config config.json" to start evaluation

Step to visualize detections :

1) Modify the "model_def" and "checkpoint_file" in the config.json

2) Type the command "python test_detection.py --config config.json" to start testing