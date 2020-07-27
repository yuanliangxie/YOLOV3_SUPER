TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet53",
        "backbone_weight": "",
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 4,
        "classes_category": ['car', 'bus', 'van', 'others']
        # "classes_category": ["car"]
    },
    "lr": {
        "backbone_lr": 1e-4,
        "other_lr": 1e-4,
        "LR_INIT": 1e-4,
        "LR_END": 1e-6,
        "WARMUP_EPOCHS": 1,
        "freeze_backbone": False,  # freeze backbone wegiths to finetune
        "decay_step": [60, 80],
        "decay_gamma": 0.1
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 5e-04,
    },
    "data_path":"/media/xyl/6418a039-786d-4cd8-b0bb-1ed36a649668/Datasets/UA-DETRAC",
    "batch_size": 8,
    "train_path": "../data/detrac/train.txt",#../data/coco/vehecal/vehecal_train.txt",
    "train_ignore_region": "../data/detrac/train_ignore_region.txt",
    "train_labels_path": "../data/detrac/labels",
    "epochs": 80,
    "Multi-scale training": True, #要增加多尺度训练！
    "img_h": 416,#如果Multi-scale training是False，则使用此单尺度训练
    "img_w": 416,
    "parallels": [0],                         #  config GPU device
    "working_dir": "/home/xyl/桌面/YOLO_SUPER",              #  replace with your working dir
    "pretrain_snapshot": "/home/xyl/PycharmProjects/YOLO_superX/darknet53/Multi-scale_trydetrac_baseline_train/20200625231341/model.pth",
    #../darknet53/Multi-scale_trydetrac_baseline_train/20200624231208/model.pth
    "try": "detrac_baseline_train",
    "scheduler_way": "Cosdecay"
}

Eval = {
        "DATA_PATH": "/home/xyl/Pycharmproject/YOLOv3/voc_data",     #voc数据集所放地址
        "PROJECT_PATH": "/home/xyl/桌面/YOLO_SUPER", #即本项目的地址
        "TEST_IMG_SIZE":544,
        "BATCH_SIZE":32,
        "NUMBER_WORKERS":0,
        "CONF_THRESH":0.01,
        "NMS_THRESH":0.5,
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False,
        "test_path": "../data/detrac/test.txt",#../data/coco/vehecal/vehecal_train.txt",
        "test_ignore_region": "../data/detrac/test_ignore_region.txt",
        "test_labels_path": "../data/detrac/labels_test",
        }
