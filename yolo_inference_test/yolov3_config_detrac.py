# coding=utf-8
# project
# test
TEST = {
	'DATA':{"CLASSES":['car'],
			"NUM":1},
	"TEST_IMG_SIZE":(640, 640),
	"CONF_THRESH": 0.9,
	"NMS_THRESH": 0.5,
	"device_id": 0, #or"cpu"or"CPU
	"model": {
		"anchors": [[[116, 90], [156, 198], [373, 326]],
					[[30, 61], [62, 45], [59, 119]],
					[[10, 13], [16, 30], [33, 23]]],
		"classes": 1,
	},
	#"pretrain_snapshot": "./weights/model_map_0.925.pth"
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try_yolov3_baseline_SSDaug_coco_pretrain_loss_div_all_objects_test_UA_detrac/20210312092005/model_map_0.986.pth",
	"ce": False,
	"bce": True,
}