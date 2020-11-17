# coding=utf-8
# project
# test
TEST = {
	"device_id": 0,
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER",
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
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try_centernet_test_UA_detrac/20201016224944/model_map_0.761.pth",

	"generate_analyze_figure": False,
	"generate_analyze_figure_dir_name":"analyze_loss_yolov3"
}

