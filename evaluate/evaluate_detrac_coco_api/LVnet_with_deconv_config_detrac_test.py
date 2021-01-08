# coding=utf-8
# project
# test
TEST = {
	"device_id": 0,
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER",
	"test_path": "../../data/detrac/test.txt",
	"test_ignore_region": "../../data/detrac/test_ignore_region.txt",
	"test_labels_path": "../../data/detrac/labels_test",
	'DATA':{"CLASSES":['car'],
			"NUM":1},
	"TEST_IMG_SIZE":640,
	"BATCH_SIZE":16,
	"NUMBER_WORKERS":0,
	"CONF_THRESH":0.01,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"model": {
		"classes": 1,
	},
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/size640x640_try_LVnet_with_deconv_test_UA_detrac/20210104222140/model.pth",

	"generate_analyze_figure": False,
	"generate_analyze_figure_dir_name":"analyze_loss_LVnet_768"
}

