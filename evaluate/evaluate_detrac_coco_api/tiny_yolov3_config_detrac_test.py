# coding=utf-8
# project
# test
TEST = {
	"device_id": 0,
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER",
	"test_path": "../../data/detrac/test.txt",
	"test_ignore_region": "../../data/detrac/test_ignore_region.txt",
	"test_labels_path": "../../data/detrac/labels_test",
	'DATA':{"CLASSES":['car'],#['car', 'bus', 'van', 'others'],
			"NUM":1},
	"TEST_IMG_SIZE":640,
	"BATCH_SIZE":32,
	"NUMBER_WORKERS":0,
	"CONF_THRESH":0.01,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"model": {
		"anchors": [[[10, 14], [23, 27], [37, 58]],
					[[81, 82], [135, 169], [344, 319]]],
		"classes": 1,
	},
	"label_smooth": False, #label_smooth还有一些问题要跟ce适应
	"GIOU": False,
	"mix_up": False,
	"ce": False,
	"bce": True,


	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/size640x640_try_tiny_yolov3_test_UA_detrac/20210116173625/model_map_0.971.pth",

	"generate_analyze_figure":False,
	"generate_analyze_figure_dir_name":"analyze_loss_yolov3_baseline_UA_detrac_test"
}

