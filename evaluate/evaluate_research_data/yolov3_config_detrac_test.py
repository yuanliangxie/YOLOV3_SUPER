TEST = {
	"device_id": 0,
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER",
	"test_path": "../../data/research_11_50_data/test.txt",
	"test_labels_path": "../../data/research_11_50_data/labels_test",
	'DATA':{"CLASSES":['car'],#, 'bus', 'van', 'others'
			"NUM":1},
	"TEST_IMG_SIZE":640,
	"BATCH_SIZE":32,
	"NUMBER_WORKERS":0,
	"CONF_THRESH":0.01,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"model": {
		"anchors": [[[116, 90], [156, 198], [373, 326]],
					[[30, 61], [62, 45], [59, 119]],
					[[10, 13], [16, 30], [33, 23]]],
		"classes": 1,
	},
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/size640x640_try_yolov3_baseline_test_UA_detrac/20210111102829/model.pth",
	#/home/xyl/PycharmProjects/YOLO_inference_count/yolo_inference/weights/model_map_0.925.pth
	"ce": False,
	"bce": True,
	"generate_analyze_figure": False,
	"generate_analyze_figure_dir_name":"analyze_ce_loss_yolo"
}