# coding=utf-8
# project
# test
TEST = {
	'DATA':{"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
					   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
					   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
					   'train', 'tvmonitor'],
			"NUM":20},
	"TEST_IMG_SIZE":(320, 608),
	"CONF_THRESH": 0.1,
	"NMS_THRESH": 0.45,
	"device_id": 0,
	"yolo": {
		"anchors": [[[116, 90], [156, 198], [373, 326]],
					[[30, 61], [62, 45], [59, 119]],
					[[10, 13], [16, 30], [33, 23]]],
		"classes": 20,
	},
	#"pretrain_snapshot": "./weights/model_map_0.925.pth"
	"pretrain_snapshot": "/media/xyl/6418a039-786d-4cd8-b0bb-1ed36a649668/训练权重文件/Multi-scale_try0/20200909221333/model_map_0.830.pth",
	"ce": False,
	"bce": True,
}