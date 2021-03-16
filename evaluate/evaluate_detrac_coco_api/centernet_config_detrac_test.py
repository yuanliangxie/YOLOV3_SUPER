# coding=utf-8
# project
# test
TEST = {
	"device_id": 0,
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER",
	"test_path": "../../data/detrac/test.txt",
	"test_ignore_region": "../../data/detrac/test_ignore_region.txt",
	"test_labels_path": "../../data/detrac/labels_test",
	'DATA':{#"CLASSES":['car', 'bus', 'van', 'others'],
			"CLASSES":['car'],
			"NUM":1},
	"TEST_IMG_SIZE":640,
	"BATCH_SIZE":32,
	"NUMBER_WORKERS":0,
	"CONF_THRESH":0.1,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"model": {
		"classes": 1,
	},
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try_centernet_test_UA_detrac/20210103214308/model_map_0.983.pth",
	#/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try_centernet_test_UA_detrac/20210103202155/model.pth
	#/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try_centernet_test_UA_detrac/20201016224944/model_map_0.761.pth
	"generate_analyze_figure": False,
	"generate_analyze_figure_dir_name":"analyze_loss_centernet_18"
}

