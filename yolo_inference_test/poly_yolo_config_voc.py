# coding=utf-8
# project
# test
TEST = {
	'DATA':{"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
					   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
					   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
					   'train', 'tvmonitor'],
			"NUM":20},
	"TEST_IMG_SIZE":(192, 320),
	"CONF_THRESH": 0.5,
	"NMS_THRESH": 0.45,
	"device_id": 0,
	"yolo": {
		"anchors": [[62, 45]],
		"classes": 20,
	},
	#"pretrain_snapshot": "./weights/model_map_0.925.pth"
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try0/20200911110311/model.pth"
}

