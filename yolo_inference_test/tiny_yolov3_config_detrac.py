# coding=utf-8
# project
# test
TEST = {
	'DATA':{"CLASSES":['car'],
			"NUM":1},
	"TEST_IMG_SIZE":(320, 320),
	"CONF_THRESH": 0.2,
	"NMS_THRESH": 0.5,
	"device_id": 'cpu',
	"model": {
		"anchors": [[[10, 14], [23, 27], [37, 58]],
					[[81, 82], [135, 169], [344, 319]]],
		"classes": 1,
	},
	#"pretrain_snapshot": "./weights/model_map_0.925.pth"
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/size640x640_try_tiny_yolov3_test_UA_detrac/20210116173625/model_map_0.971.pth",
	"ce": False,
	"bce": True,

}

