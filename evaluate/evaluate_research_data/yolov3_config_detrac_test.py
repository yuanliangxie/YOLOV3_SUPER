TEST = {
	"device_id": 0,
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER",
	"test_path": "../../data/research_11_50_data/test.txt",
	"test_labels_path": "../../data/research_11_50_data/labels_test",
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
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLO_inference_count/yolo_inference/weights/model_map_0.925.pth",
	#/home/xyl/PycharmProjects/YOLO_inference_count/yolo_inference/weights/model_map_0.925.pth
	"ce": False,
	"bce": True,
	"generate_analyze_figure": False,
	"generate_analyze_figure_dir_name":"analyze_ce_loss_yolo"
}