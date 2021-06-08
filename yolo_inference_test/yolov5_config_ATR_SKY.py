# coding=utf-8
# project
# test
TEST = {
	'DATA':{"CLASSES":['car'],
			"NUM":1},
	"TEST_IMG_SIZE":(640, 480),
	"CONF_THRESH": 0.11,
	"NMS_THRESH": 0.5,
	"device_id": 0, #or"cpu"or"CPU
	"model": {
		"anchors": [[[116, 90], [156, 198], [373, 326]],
					[[30, 61], [62, 45], [59, 119]],
					[[10, 13], [16, 30], [33, 23]]],
		"classes": 1,
	},
	#"pretrain_snapshot": "./weights/model_map_0.925.pth"
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/CSPdarknet53/Multi-scale_try_yolov5_baseline_SSDaug_ATR_sky/20210604222902/model_map_0.853.pth",
	"ce": False,
	"bce": True,
}