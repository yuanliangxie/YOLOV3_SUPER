TRAINING_PARAMS = \
	{
		"model_params": {
			"backbone_name": "Centernet_resnet18",
			"backbone_weight": "",
		},
		"model": {
			"classes": 1,
			#"classes_category": ['car', 'bus', 'van', 'others']
			"classes_category": ["car"]
		},
		"lr": {
			"backbone_lr": 1e-3,
			"other_lr": 1e-3,
			"LR_INIT": 1e-3,
			"LR_END": 1e-5,
			"WARMUP_EPOCHS": 1,
			"freeze_backbone": False,  # freeze backbone wegiths to finetune
			"decay_step": [15, 20],
			"decay_gamma": 0.1
		},
		"optimizer": {
			"type": "sgd",
			"weight_decay": 5e-04,
		},
		"batch_size": 16,
		"train_path": "../data/ATR_sky/trainval.txt",
		"train_labels_path": "../data/ATR_sky/labels",
		"epochs": 72,
		"Multi-scale training": True, #要增加多尺度训练！
		"img_h": 608,#如果Multi-scale training是False，则使用此单尺度训练
		"img_w": 608,
		"parallels": [0],                         #  config GPU device
		"working_dir": "/home/xyl/PycharmProjects/YOLOV3_SUPER",              #  replace with your working dir

		# restore_model_weight:
		"pretrain_snapshot": "",
		# /home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/Multi-scale_try+ce_loss/20201006121114/model.pth
		"self_train_weight": True,
		"resume_start_epoch": 2,


		# train_eval:
		"start_eval": 20,
		"interval_epoch_eval": 5, #每隔多少个epoch进行验证
		"epoch_eval_times": 1, #每个epoch验证多少次
		#train_eval参数的含义为：从"start_eval"第2个epoch开始进行验证，此时"epoch_eval_times"第２个epoch总共
		# 会验证两次，然后间隔"interval_epoch_eval"２个epoch会再次进行验证


		#tricks
		"try": '_centernet_test_ATR_data',
		"scheduler_way": "Cosdecay",
		"GIOU": False,
		"mix_up": False,
		"accumulate":1
	}

Eval = {
	#"DATA_PATH": "/home/xyl/Pycharmproject/YOLOv3/voc_data",     #voc数据集所放地址
	"PROJECT_PATH": "/home/xyl/PycharmProjects/YOLOV3_SUPER", #即本项目的地址
	"TEST_IMG_SIZE":640,
	"BATCH_SIZE":32,
	"NUMBER_WORKERS":0,
	"CONF_THRESH":0.01,
	"NMS_THRESH":0.5,
	"MULTI_SCALE_TEST":False,
	"FLIP_TEST":False,
	"test_path": "../data/ATR_sky/test.txt",#../data/coco/vehecal/vehecal_train.txt",
	"test_labels_path": "../data/ATR_sky/labels_test",

	#不产生结果分析图
	"generate_analyze_figure": False,
}
