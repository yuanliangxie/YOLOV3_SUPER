# coding=utf-8
# project
# test
TEST = {
	"device_id": 0,
	#"DATA_PATH": "/media/xyl/6418a039-786d-4cd8-b0bb-1ed36a649668/Datasets/课题组视频检测标注/课题组1150视频-精标2d检测框",
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER",
	"test_path": "../../data/research_11_50_data/test.txt",
	"test_labels_path": "../../data/research_11_50_data/labels_test",
	'DATA':{#"CLASSES":['car', 'bus', 'van', 'others'],
			"CLASSES":['car'],
			"NUM":1},
	"TEST_IMG_SIZE":640,
	"BATCH_SIZE":32,
	"NUMBER_WORKERS":0,
	"CONF_THRESH":0.01,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"model": {
		#"anchors": [[62, 45]],
		"classes": 1,
	},
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try_centernet_test_UA_detrac/20210103214308/model_map_0.983.pth",
	"ce": False,
	"bce": True,
	"generate_analyze_figure": True,
	"generate_analyze_figure_dir_name":"analyze_loss_centernet_resnet18"
}

