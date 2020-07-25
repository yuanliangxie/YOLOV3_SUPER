import sys
import os
sys.path.append("../../YOLOV3_SUPER")
import time
import torch
from utils.timer import Timer
from utils.logger import log_recorder
from tensorboardX import SummaryWriter
from utils.utils_select_device import select_device
from train.config_factory import get_config
from utils.train_utils import write_params_into_logger, create_sub_workdir, get_optimizer, get_lr_scheduler, save_checkpoint
from models.model.model_factory import load_model
from dataset.dataset_factory import load_dataset
from evaluate.coco_evaluater_factory import load_coco_evaluater
from train.train_eval import train_evaler

class trainer():
	def __init__(self, config_train, config_eval, logger):
		self.config_train = config_train
		self.config_eval = config_eval

		#以下这段代码为将config_train的信息补充到config_eval上，后面coco_evaluater要用
		self.config_eval['DATA'] = {}
		self.config_eval['DATA']["CLASSES"] = self.config_train['yolo']['classes_category']
		self.config_eval['DATA']["NUM"] = self.config_train['yolo']['classes']

		self.logger = logger
		self.device = select_device(config_train['device_id'])
		self.write_params()
		self.init_params()
		self.net = self.get_model()
		self.optimizer = get_optimizer(self.config_train, self.net, self.logger)
		self.dataset = self.get_dataset()
		self.dataloader = torch.utils.data.DataLoader(self.dataset,
													  batch_size=self.config_train["batch_size"],
													  shuffle=True, num_workers=8, collate_fn=self.dataset.collate_fn, pin_memory=True, drop_last=True)
		logger.append("Total images: {}".format(len(self.dataset)))
		self.timer = Timer(len(self.dataloader), self.config_train["epochs"])
		self.lr_scheduler = get_lr_scheduler(self.config_train, self.optimizer, len(self.dataloader))
		self.restore_model()
		self.coco_evaluater = self.get_coco_evaluater()
		self.train_evaler = train_evaler(model=self.net, logger=self.logger,
					 config_eval=self.config_eval, coco_evaluater=self.coco_evaluater)


	@classmethod
	def set_config(cls, config_name, device_id, config_model_name):
		"""
		:param config_char: 选择哪个数据集对应的配置文件
		:param device_id: 选择那个gpu运行程序
		:param config_model_name: 选择那个目标检测模型
		:在这个函数中可以根据parase解析参数加入到config_train中
		:return: 训练类实例
		"""
		config_train, config_eval = get_config(config_name)
		config_train["config_name"] = config_name
		config_train["model_name"] = config_model_name
		config_train["device_id"] = device_id#TODO:这个是自己加的，可以在解析parase的时候进行加载
		#config_train["batch_size"] *= len(config_train["parallels"])

		# Create sub_working_dir
		config_train = create_sub_workdir(config_train)
		sub_working_dir = config_train['sub_working_dir']
		logger = log_recorder(os.path.join(sub_working_dir, 'log.txt'))
		logger.append(("sub working dir: %s" % sub_working_dir))

		# Creat tf_summary writer
		config_train["tensorboard_writer"] = SummaryWriter(sub_working_dir)
		# print(("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir)))
		logger.append(("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir)))

		# Start training
		os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config_train["parallels"]))
		return cls(config_train, config_eval, logger)

	def write_params(self):
		write_params_into_logger(self.config_train, self.logger, info="Training_params:")#写入训练的参数信息
		write_params_into_logger(self.config_eval, self.logger, info="Eval_params:")#写入验证的参数信息

	def init_params(self):
		self.config_train["global_step"] = self.config_train.get("start_step", 0)
		self.max_map = 0
		self.epoch_init=-1
		if self.config_train["Multi-scale training"]:
			self.image_size_train = None
		else:
			self.image_size_train = (self.config_train["img_w"], self.config_train["img_h"])

	def get_model(self):
		model = load_model(self.config_train["model_name"])
		net = model(self.config_train, self.logger)
		if self.config_train["model_params"]["backbone_weight"].split(".")[-1].strip() == "pth":
			net.load_darknet_pth_weights(self.config_train["model_params"]["backbone_weight"])
		elif len(self.config_train["model_params"]["backbone_weight"].strip()) != 0:
			#默认加载darknet的原始权重
			net.load_darknet_weights(self.config_train["model_params"]["backbone_weight"])
		net.train(True)
		net = net.to(self.device)
		return net

	def get_dataset(self):
		dataset = load_dataset(self.config_train, image_size_train=self.image_size_train)
		assert dataset != None, "数据工厂没有您所想要加载的数据"
		return dataset

	def get_coco_evaluater(self):
		coco_evaluater = load_coco_evaluater(self.config_train["config_name"])
		return coco_evaluater

	def restore_model(self):
		if self.config_train["pretrain_snapshot"].strip():
			self.logger.append("Load pretrained weights from {}".format(self.config_train["pretrain_snapshot"]))
			#print("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
			if self.config_train["pretrain_snapshot"].split('/')[-1].strip(" ") == "model.pth":
				self.max_map = 0
			else:
				self.max_map = float(self.config_train["pretrain_snapshot"][-9:-4])
			#加载最新的保存有
			load_checkpoint = torch.load(self.config_train["pretrain_snapshot"])
			state_dict = load_checkpoint["state_dict"]
			self.epoch_init = load_checkpoint["epoch"]
			self.config_train["global_step"] = load_checkpoint['global_step']
			self.net.load_state_dict(state_dict)
			self.optimizer.load_state_dict(load_checkpoint['optimizer'])
			self.lr_scheduler.step(self.config_train["global_step"])#通过得到已经走过的步数信息，更新lr
			self.config_train["global_step"] += 1

	def start_train(self):
		# Start the training loop
		self.logger.append("Start training.")
		#print("Start training.")
		for epoch in range(self.epoch_init+1, self.config_train["epochs"]):
			for step, samples in enumerate(self.dataloader):

				images, labels = samples["image"].to(self.device), samples["label"]
				start_time = time.time()


				# Forward and backward
				self.optimizer.zero_grad()
				outputs = self.net(images, target=labels)
				losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
				losses = []
				for _ in range(len(losses_name)):
					losses.append([])
				for i in range(3):
					for j, l in enumerate(outputs[i]):
						losses[j].append(l)
				losses = [sum(l) for l in losses]#TODO:这里用sum会进行反向传播吗？经过简单的实验验证是可以反向传播的！
				loss = losses[0]  # 求的是total_loss
				loss.backward()
				self.optimizer.step()
				self.lr_scheduler.step(self.config_train["global_step"])#这里两个Ir_scheduler合并一致了

				if step >0 and step % int(len(self.dataloader)-1) == 0:#TODO:设置定时放map，此时是一个epoch来记录一下模型

					save_checkpoint(self.net.state_dict(), self.config_train, map=None, optimizer=self.optimizer, epoch=epoch, logger=self.logger)#TODO：记录最佳的map与正常的模型，存储模型时加入epoch和step信息，便于重新加载训练！

					if epoch >= 0 and epoch % 5 == 0:
						images.cpu()
						del samples
						self.net.train(False)  # 进入eval模式
						self.train_evaler.model = self.net
						map = self.train_evaler.eval_voc()
						self.net.train(True)  # 再次开启训练模式
						if map > self.max_map:
							if os.path.exists(os.path.join(self.config_train["sub_working_dir"], "model_map_%.3f.pth" % self.max_map)):
								os.remove(os.path.join(self.config_train["sub_working_dir"], "model_map_%.3f.pth" % self.max_map))
							self.max_map = map
							save_checkpoint(self.net.state_dict(), self.config_train, map, self.optimizer, epoch, self.logger)
						self.config_train['tensorboard_writer'].add_scalar('map', map, epoch)


				if step > 0 and step % 10 == 0:
					_loss = loss.item()
					duration = float(time.time() - start_time)#TODO：可以考虑加入估计停止时间
					self.timer.add(duration)
					remain_time = self.timer.get_remain_time(step+epoch*len(self.dataloader))
					example_per_second = self.config_train["batch_size"] / duration
					lr = self.optimizer.param_groups[0]['lr']
					self.logger.append(
						"epoch [%.3d] iter = %d loss = %.5f batch/sec = %.3f lr = %.7f remainTime = %s"%
						(epoch, step, _loss, example_per_second, lr, remain_time)
					)
					self.config_train["tensorboard_writer"].add_scalar("lr",
															lr,
															self.config_train["global_step"])
					self.config_train["tensorboard_writer"].add_scalar("example/sec",
															example_per_second,
															self.config_train["global_step"])
					for i, name in enumerate(losses_name):
						value = _loss if i == 0 else losses[i]
						self.config_train["tensorboard_writer"].add_scalar(name,
																value,
																self.config_train["global_step"])

				self.config_train["global_step"] += 1
		self.logger.append("Bye~")


if __name__ == "__main__":
	#在跑程序前需要清空../evaluate/data或者../evaluate_coco/data或者evaluate_detrac_coco_api方法中的文件
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_name', type=str, default='VOC', help='VOC or U-DETRAC')
	parser.add_argument('--device_id', type=int, default=0, help="choose the device_id")
	parser.add_argument('--config_model_name', type=str, default='yolov3', help='you can cd ./models/model/model_factory to find model name')
	opt = parser.parse_args()
	trainer_voc = trainer.set_config(opt.config_name, opt.device_id, opt.config_model_name)
	trainer_voc.start_train()
