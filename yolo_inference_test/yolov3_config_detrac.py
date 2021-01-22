# coding=utf-8
# project
# test
TEST = {
	'DATA':{"CLASSES":['car'],
			"NUM":1},
	"TEST_IMG_SIZE":(320, 320),
	"CONF_THRESH": 0.1,
	"NMS_THRESH": 0.45,
	"device_id": 0,
	"model": {
		"anchors": [[[116, 90], [156, 198], [373, 326]],
					[[30, 61], [62, 45], [59, 119]],
					[[10, 13], [16, 30], [33, 23]]],
		"classes": 1,
	},
	#"pretrain_snapshot": "./weights/model_map_0.925.pth"
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/size640x640_try_yolov3_baseline_test_UA_detrac/20210111102829/model.pth",
	"ce": False,
	"bce": True,
}