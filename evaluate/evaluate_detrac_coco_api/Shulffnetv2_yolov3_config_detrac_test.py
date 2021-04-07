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
	"CONF_THRESH":0.1,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"model": {
		"anchors": [[[116, 90], [156, 198], [373, 326]],
					[[30, 61], [62, 45], [59, 119]],
					[[10, 13], [16, 30], [33, 23]]],
		"classes": 1,
	},
	"label_smooth": False, #label_smooth还有一些问题要跟ce适应
	"GIOU": False,
	"mix_up": False,
	"ce": False,
	"bce": True,


	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try_yolov3_shulffnetv2_SSDaug_coco_pretrain_loss_div_all_objects_test_UA_detrac/20210327191207/model_map_0.983.pth",
	#/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/size640x640_try_yolov3_baseline_test_UA_detrac/20210111102829/model.pth
	"generate_analyze_figure":False,
	"generate_analyze_figure_dir_name":"analyze_loss_shulffnetv2_UA_detrac_test"
}

