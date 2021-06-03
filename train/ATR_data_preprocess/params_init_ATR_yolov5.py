TRAINING_PARAMS = \
	{
		"model_params": {
			"backbone_name": "CSPdarknet53",
			"backbone_weight": "",
		},
		"model": {
			"anchors": [[[116, 90], [156, 198], [373, 326]],
						[[30, 61], [62, 45], [59, 119]],
						[[10, 13], [16, 30], [33, 23]]],
			"classes": 1,
			"classes_category": ['car']
			# "classes_category": ["car"]
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
		#"data_path":"/media/xyl/6418a039-786d-4cd8-b0bb-1ed36a649668/Datasets/YOLO_SUPER_DATASET/voc_data",
		"batch_size": 16,
		"train_path": "../data/ATR_sky/trainval.txt",#../data/coco/vehecal/vehecal_train.txt",
		"train_labels_path": "../data/ATR_sky/labels",
		"epochs": 71,
		"Multi-scale training": True, #要增加多尺度训练！
		"img_h": 640, #只有在单尺度下，这个尺寸才会生效！
		"img_w": 640,
		"parallels": [0, 1],                         #  config GPU device
		"working_dir": "/home/xyl/PycharmProjects/YOLOV3_SUPER",              #  replace with your working dir

		#restore_model_weight:
		"pretrain_snapshot": "",
		"self_train_weight": True, #加载由此框架训练得到的权重则设置为True,否则设置为False
		"resume_start_epoch": None,

		# train_eval:
		"start_eval": 75,
		"interval_epoch_eval": 5, #每隔多少个epoch进行验证
		"epoch_eval_times": 1, #每个epoch验证多少次
		#train_eval参数的含义为：从"start_eval"第１０个epoch开始进行验证，此时第１０个epoch总共
		# 会验证"epoch_eval_times"１次，然后间隔"interval_epoch_eval"５个epoch会再次进行验证


		#tricks
		"try": '_yolov5_baseline_SSDaug_ATR_sky',
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
	"BATCH_SIZE":16,
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
