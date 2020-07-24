# coding=utf-8
# project
# test
TEST = {
        "device_id":0,
        "PROJECT_PATH": "/home/xyl/桌面/YOLO_SUPER",
        "test_path": "../../data/detrac/test.txt",
        "test_ignore_region": "../../data/detrac/test_ignore_region.txt",
        "test_labels_path": "../../data/detrac/labels_test",
        'DATA':{"CLASSES":['car', 'bus', 'van', 'others'],
                "NUM":4},
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
                "classes": 4,
                },
        "pretrain_snapshot": "/home/xyl/PycharmProjects/YOLO_superX/darknet53/Multi-scale_trydetrac_baseline_train/20200625231341/model.pth"
        }

