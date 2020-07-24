import os
import time
import torch
import torch.optim as optim
from utils.schedule import CosineDecayLR, StepDecayLR

def write_params_into_logger(config, logger, info):
	logger.append(' ')
	logger.append(info)
	for key in config:
		logger.append(str(key)+':'+str(config[key]))

def create_sub_workdir(config):
	if config['Multi-scale training'] == True:
		strings = 'Multi-scale'
	else:
		strings = 'size{}x{}'.format(config['img_w'], config['img_h'])
	sub_working_dir = '{}/{}/{}_try{}/{}'.format(
		config['working_dir'], config['model_params']['backbone_name'],
		strings, config['try'],
		time.strftime("%Y%m%d%H%M%S", time.localtime()))
	if not os.path.exists(sub_working_dir):
		os.makedirs(sub_working_dir)
	config["sub_working_dir"] = sub_working_dir
	return config

def save_checkpoint(state_dict, config, map, optimizer, epoch, logger):
	# global best_eval_result
	if map == None:
		checkpoint_path = os.path.join(config["sub_working_dir"], "model.pth")
		checkpoint = {'state_dict': state_dict,
					  "global_step": config["global_step"],
					  'optimizer': optimizer.state_dict(),
					  'epoch': epoch}
		torch.save(checkpoint, checkpoint_path)
		logger.append("Model checkpoint saved to %s" % checkpoint_path)
	else:
		checkpoint_path = os.path.join(config["sub_working_dir"], "model_map_%.3f.pth"%map)
		checkpoint = {'state_dict': state_dict,
					  "global_step": config["global_step"],
					  'optimizer': optimizer.state_dict(),
					  'epoch': epoch}
		torch.save(checkpoint, checkpoint_path)
		logger.append("Best Model checkpoint saved to %s" % checkpoint_path)

def get_optimizer(config, net, logger):
	optimizer = None

	# Assign different lr for each layer
	params = None
	base_params = list(
		map(id, net.backbone.parameters())
	)
	logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

	if not config["lr"]["freeze_backbone"]:
		params = [
			{"params": logits_params, "lr": config["lr"]["other_lr"]},
			{"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
		]
	else:
		logger.append("freeze backbone's parameters.")
		#print("freeze backbone's parameters.")
		for p in net.backbone.parameters():
			p.requires_grad = False
		params = [
			{"params": logits_params, "lr": config["lr"]["other_lr"]},
		]

	# Initialize optimizer class
	if config["optimizer"]["type"] == "adam":
		logger.append("Using adam optimizer.")
		optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])

	elif config["optimizer"]["type"] == "amsgrad":
		optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
							   amsgrad=True)
	elif config["optimizer"]["type"] == "rmsprop":
		optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])

	else:
		# Default to sgd
		logger.append("Using SGD optimizer.")
		#print("Using SGD optimizer.")
		optimizer = optim.SGD(params, momentum=0.9,
							  weight_decay=config["optimizer"]["weight_decay"],
							  nesterov=(config["optimizer"]["type"] == "nesterov"))

	return optimizer

def get_lr_scheduler(config_train, optimizer, len_dataloader):
	"""
	:param config_train:
	:param optimizer:
	:param len_dataloader: == len(self.dataloader)
	:return:
	"""
	if config_train['scheduler_way'] == 'Cosdecay':
		lr_scheduler = CosineDecayLR(optimizer,
									 T_max=config_train['epochs'] * len_dataloader,
									 lr_init=config_train['lr']["LR_INIT"],
									 lr_min=config_train['lr']["LR_END"],
									 warmup=config_train['lr']["WARMUP_EPOCHS"] * len_dataloader)
	elif config_train['scheduler_way'] == 'Stepdecay':
		lr_scheduler = StepDecayLR(optimizer, T_max=config_train['epochs'] * len_dataloader,
								   lr_init=config_train['lr']["LR_INIT"],
								   gamma=config_train["lr"]["decay_gamma"],
								   decay_step=[i * len_dataloader for i in config_train['lr']['decay_step']])
	return lr_scheduler