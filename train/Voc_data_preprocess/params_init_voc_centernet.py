TRAINING_PARAMS = \
	{
		"model_params": {
			"backbone_name": "darknet53",
			"backbone_weight": "",
		},
		"yolo": {
			"classes": 20,
			"classes_category": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
								 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
								 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
								 'train', 'tvmonitor']
			# "classes_category": ["car"]
		},
		"lr": {
			"backbone_lr": 1e-3,
			"other_lr": 1e-3,
			"LR_INIT": 1e-3,
			"LR_END": 1e-5,
			"WARMUP_EPOCHS": 1,
			"freeze_backbone": False,  # freeze backbone wegiths to finetune
			"decay_step": [60, 80],
			"decay_gamma": 0.1
		},
		"optimizer": {
			"type": "sgd",
			"weight_decay": 5e-04,
		},
		"data_path":"/home/xyl/Pycharmproject/YOLOv3/voc_data",
		"batch_size": 16,
		"train_path": "../data/voc/trainval.txt",#../data/coco/vehecal/vehecal_train.txt",
		"train_labels_path": "../data/voc/labels",
		"epochs": 80,
		"Multi-scale training": True, #要增加多尺度训练！
		"img_h": 608, #只有在单尺度下，这个尺寸才会生效！
		"img_w": 608,
		"parallels": [0, 1],                         #  config GPU device
		"working_dir": "/home/xyl/PycharmProjects/YOLOV3_SUPER",              #  replace with your working dir

		# restore_model_weight:
		"pretrain_snapshot": "",
		# /home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try+ce_loss/20201006121114/model.pth
		# /home/xyl/桌面/YOLO_SUPER/darknet53/Multi-scale_try0/20200723120846/model.pth
		# /home/xyl/PycharmProjects/YOLOV3_baseline/darknet53/Multi-scale_try0/20200523150149/model_map_0.835.pth
		# ../darknet53/Multi-scale_try0/20200522220233/model_map_0.812.pth
		"self_train_weight": True,
		"resume_start_epoch": 1,


		# train_eval:
		"start_eval": 0,
		"interval_epoch_eval": 5, #每隔多少个epoch进行验证
		"epoch_eval_times": 20, #每个epoch验证多少次
		#train_eval参数的含义为：从"start_eval"第１０个epoch开始进行验证，此时第１０个epoch总共
		# 会验证"epoch_eval_times"１次，然后间隔"interval_epoch_eval"５个epoch会再次进行验证

		#tricks
		"try": '_centernet_test',
		"scheduler_way": "Cosdecay",
		"GIOU": False,
		"mix_up": False,
		"accumulate":1
	}

Eval = {
	"DATA_PATH": "/home/xyl/Pycharmproject/YOLOv3/voc_data",     #voc数据集所放地址
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER", #即本项目的地址
	"TEST_IMG_SIZE":544,
	"BATCH_SIZE":32,
	"NUMBER_WORKERS":0,
	"CONF_THRESH":0.01,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"test_path": "../data/voc/test.txt",
	"test_labels_path": "../data/voc/labels_test",

	#不产生结果分析图
	"generate_analyze_figure":False,
}
