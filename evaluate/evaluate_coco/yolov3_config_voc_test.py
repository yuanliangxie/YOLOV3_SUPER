# coding=utf-8
# project
# test
TEST = {
        "device_id": 0,
        "DATA_PATH": "/home/xyl/Pycharmproject/YOLOv3/voc_data",
        "PROJECT_PATH": "/home/xyl/桌面/YOLO_SUPER",
        "test_path": "../../data/voc/test.txt",
        "test_labels_path": "../../data/voc/labels_test",
        'DATA':{"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                   'train', 'tvmonitor'],
                "NUM":20},
        "TEST_IMG_SIZE":544,
        "BATCH_SIZE":32,
        "NUMBER_WORKERS":0,
        "CONF_THRESH":0.01,
        "NMS_THRESH":0.5,
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False,
        "yolo": {
                "anchors": [[[116, 90], [156, 198], [373, 326]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[10, 13], [16, 30], [33, 23]]],
                "classes": 20,
                },
        "pretrain_snapshot": "/home/xyl/桌面/YOLO_SUPER/darknet53/Multi-scale_try0/20200723220121/model.pth"
        }

