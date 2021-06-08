import torch.nn as nn
from models.backbone.centernet_hourglass.hourglass import exkp
from models.bricks.centernet_bricks import Conv2d, DeConv2d, SPP
from models.loss.centernet_hourglass_loss_module import centernet_loss_module
from utils.logger import print_logger
import numpy as np
import torch
from utils.utils_select_device import select_device

class centernet_hourglass(nn.Module):
	def __init__(self, config, logger=None, init_weight=True, nstack=2):
		super().__init__()
		self.backbone = exkp(n=5, nstack=nstack, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4], num_classes=config['model']['classes'])
		self.loss = centernet_loss_module(config, stride=4, nstack=nstack)

		if logger == None:
			self.logger = print_logger()
		else:
			self.logger = logger

		if init_weight:
			self.__init_weights()


		# self.cls_pred = nn.Sequential(
		# 	Conv2d(256, 64, ksize=3, padding=1, leakyReLU=True),
		# 	nn.Conv2d(64, self.num_classes, kernel_size=1)
		# )
		#
		# self.txty_pred = nn.Sequential(
		# 	Conv2d(256, 64, ksize=3, padding=1, leakyReLU=True),
		# 	nn.Conv2d(64, 2, kernel_size=1)
		# )
		#
		# self.twth_pred = nn.Sequential(
		# 	Conv2d(256, 64, ksize=3, padding=1, leakyReLU=True),
		# 	nn.Conv2d(64, 2, kernel_size=1)
		# )

	def __init_weights(self):

		" Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
		for m in self.modules():#
			if isinstance(m, nn.Conv2d):
				torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
				# torch.nn.init.constant_(m.weight.data,0.001)#在测试时为了看模型有没有弄错，进行的改动
				if m.bias is not None:
					m.bias.data.zero_()
				print("initing {}".format(m))

			elif isinstance(m, nn.BatchNorm2d):
				torch.nn.init.constant_(m.weight.data, 1.0)
				torch.nn.init.constant_(m.bias.data, 0.0)

				print("initing {}".format(m))


	def forward(self, input, target=None):

		outputs = self.backbone(input)
		loss_or_output = self.loss(outputs, target)

		return loss_or_output #input=416,[13, 26, 52]

if __name__ == '__main__':
	#计算参数量和计算量
	from thop import profile
	from thop import clever_format
	config={"device_id":'cpu', "num_classes":1, "model":{"classes":1}}
	model = centernet_hourglass(config, nstack=2)
	input = torch.randn(1, 3, 640, 640)
	output = model(input)
	a = 1

	# flops, params = profile(model, inputs=(input, ))
	# flops, params = clever_format([flops, params], "%.3f")# 增加可读性
	# print(flops, params)



