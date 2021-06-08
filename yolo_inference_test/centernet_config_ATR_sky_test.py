# coding=utf-8
# project
# test
TEST = {
	"device_id": 0,
	#"DATA_PATH": "/home/xyl/Pycharmproject/YOLOv3/voc_data",
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER",
	"test_path": "../../data/ATR_sky/test.txt",
	"test_labels_path": "../../data/ATR_sky/labels_test",
	'DATA':{"CLASSES":['car'],
			"NUM":1},
	"TEST_IMG_SIZE":(640,480),
	"BATCH_SIZE":32,
	"NUMBER_WORKERS":0,
	"CONF_THRESH":0.11,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"model": {
		"anchors": [[[116, 90], [156, 198], [373, 326]],
					[[30, 61], [62, 45], [59, 119]],
					[[10, 13], [16, 30], [33, 23]]],
		"classes": 1,
	},
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/Centernet_resnet18/Multi-scale_try_centernet_test_ATR_data/20210604130153/model_map_0.860.pth",
	"ce": False,
	"bce": True,
	"generate_analyze_figure": True,
	"generate_analyze_figure_dir_name":"analyze_centernet"
}


