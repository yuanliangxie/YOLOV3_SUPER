# coding=utf-8
# project
# test
TEST = {
	"device_id": 0,
	"DATA_PATH": "/home/xyl/Pycharmproject/YOLOv3/voc_data",
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER",
	"test_path": "../../data/voc/test.txt",
	"test_labels_path": "../../data/voc/labels_test",
	'DATA':{"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
					   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
					   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
					   'train', 'tvmonitor'],
			"NUM":20},
	"TEST_IMG_SIZE":544,
	"BATCH_SIZE":32,
	"NUMBER_WORKERS":0,
	"CONF_THRESH":0.01,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"yolo": {
		"anchors": [[62, 45]],
		"classes": 20,
	},
	"pretrain_snapshot": "/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try_poly_yolo_test/20201014161130/model.pth",
	"ce": False,
	"bce": True,
	"generate_analyze_figure": False,
	"generate_analyze_figure_dir_name":"analyze_loss_poly_yolo"
}

