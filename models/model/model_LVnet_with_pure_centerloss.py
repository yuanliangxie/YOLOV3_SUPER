import torch.nn as nn
from models.backbone.LVnet.LV_net import LV_Net_backbone as backbone
from models.head.LVnet_head import LVnetHead as head
from models.backbone.LVnet.LV_net import Conv2dBatchRelu6
from models.bricks.centernet_bricks import SPP
from models.loss.centernet_loss_module import centernet_loss_module as loss
from utils.logger import print_logger
import numpy as np
import torch
from utils.utils_select_device import select_device
import math
class DeConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, ksize, stride=2, leakyReLU=False):
		super(DeConv2d, self).__init__()
		# deconv basic config
		if ksize == 4:
			padding = 1
			output_padding = 0
		elif ksize == 3:
			padding = 1
			output_padding = 1
		elif ksize == 2:
			padding = 0
			output_padding = 0

		self.convs = nn.Sequential(
			nn.ConvTranspose2d(in_channels, out_channels, ksize, stride=stride, padding=padding, output_padding=output_padding),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU6(inplace=True)
		)

	def forward(self, x):
		return self.convs(x)

class LVnet(nn.Module):
	def __init__(self, config, logger=None, init_weight=True):
		super().__init__()
		self.backbone = backbone()
		#self.neck = neck()
		self.head1 = head(in_channels=128, out_channels=64, nClass=config["model"]["classes"])

		self.smooth = nn.Sequential(
			SPP(),
			Conv2dBatchRelu6(1024, 128, kernel_size=1, stride=1, padding=0),
			Conv2dBatchRelu6(128, 256, kernel_size=3, stride=1, padding=1)
		)

		self.deconv4 = DeConv2d(256, 128, ksize=4, stride=2)
		self.deconv3 = DeConv2d(128, 64, ksize=4, stride=2)
		self.deconv2 = DeConv2d(64, 128, ksize=4, stride=2)

		self.loss = loss(config)
		if logger == None:
			self.logger = print_logger()
		else:
			self.logger = logger
		if init_weight:
			self.__init_weights()

	def forward(self, input, target=None):
		features = self.backbone(input)
		#neck_features = self.neck(features)
		f1, f2, f3, f4 = features
		f4 = self.smooth(f4)
		f3 = self.deconv4(f4) #尝试用加法试试
		f2 = self.deconv3(f3)
		f1 = self.deconv2(f2)
		head1 = self.head1(f1)
		loss_or_output = self.loss(head1, target)
		return loss_or_output #input=416,[13, 26, 52]

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
	# load_checkpoint = torch.load("/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/size640x640_try_LVnet_test_UA_detrac/20201228114516/model_map_0.917.pth")
	# state_dict = load_checkpoint["state_dict"]
	# state_dict = {k[9:]:v  for k, v in state_dict.items() if "backbone" in k}
	# print(self.backbone.load_state_dict(state_dict, strict=True))
	# print("已加载训练后的权重")


if __name__ == '__main__':
	#计算参数量和计算量
	from thop import profile
	from thop import clever_format
	config={"device_id":'cpu', "num_classes":1, "model":{"classes":1}}
	model = LVnet(config)
	input = torch.randn(1, 3, 640, 640)
	flops, params = profile(model, inputs=(input, ))
	flops, params = clever_format([flops, params], "%.3f")# 增加可读性
	print(flops, params)

