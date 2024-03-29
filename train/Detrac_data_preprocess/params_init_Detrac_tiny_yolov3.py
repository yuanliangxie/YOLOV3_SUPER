TRAINING_PARAMS = \
	{
		"model_params": {
			"backbone_name": "darknet53",
			"backbone_weight": "",
			#../weights/darknet53_weights_pytorch.pth
		},
		"model": {
			"anchors": [[[10, 14], [23, 27], [37, 58]],
						[[81, 82], [135, 169], [344, 319]]],
			"classes": 1,
			#"classes_category": ['car', 'bus', 'van', 'others']
			"classes_category": ["car"]
		},
		"lr": {
			"backbone_lr": 1e-4,
			"other_lr": 1e-4,
			"LR_INIT": 1e-4,
			"LR_END": 1e-6,
			"WARMUP_EPOCHS": 1,
			"freeze_backbone": False,  # freeze backbone wegiths to finetune
			"decay_step": [60, 80],
			"decay_gamma": 0.1
		},
		"optimizer": {
			"type": "sgd",
			"weight_decay": 5e-04,
		},
		"data_path":"/media/xyl/6418a039-786d-4cd8-b0bb-1ed36a649668/Datasets/UA-DETRAC",
		"batch_size": 16,
		"train_path": "../data/detrac/train.txt",#../data/coco/vehecal/vehecal_train.txt",
		"train_ignore_region": "../data/detrac/train_ignore_region.txt",
		"train_labels_path": "../data/detrac/labels",
		"epochs": 32,
		"Multi-scale training": False, #要增加多尺度训练！
		"img_h": 640,#如果Multi-scale training是False，则使用此单尺度训练
		"img_w": 640,
		"parallels": [0],                         #  config GPU device
		"working_dir": "/home/xyl/PycharmProjects/YOLOV3_SUPER",              #  replace with your working dir

		# restore_model_weight:
		"pretrain_snapshot": "",
		# /home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try_LFFD_test_UA_detrac/20201118222434/model.pth
		"self_train_weight": True,
		"resume_start_epoch": 10,


		# train_eval:
		"start_eval": 31,
		"interval_epoch_eval": 1, #每隔多少个epoch进行验证
		"epoch_eval_times": 1, #每个epoch验证多少轮
		#train_eval参数的含义为：从"start_eval"第2个epoch开始进行验证，此时"epoch_eval_times"第２个epoch总共
		# 会验证两次，然后间隔"interval_epoch_eval"２个epoch会再次进行验证


		#tricks
		"try": '_tiny_yolov3_test_UA_detrac',
		"scheduler_way": "Cosdecay",
		"label_smooth": False, #label_smooth还有一些问题要跟ce适应
		"GIOU": False,
		"mix_up": False,
		"ce": False,
		"bce": True,
		"accumulate":1
	}

Eval = {
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER", #即本项目的地址
	"TEST_IMG_SIZE":640,
	"BATCH_SIZE":32,
	"NUMBER_WORKERS":0,
	"CONF_THRESH":0.01,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"test_path": "../data/detrac/test.txt",#../data/coco/vehecal/vehecal_train.txt",
	"test_ignore_region": "../data/detrac/test_ignore_region.txt",
	"test_labels_path": "../data/detrac/labels_test",

	#不产生结果分析图
	"generate_analyze_figure": False,
}
