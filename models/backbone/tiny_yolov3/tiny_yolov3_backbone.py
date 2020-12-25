import torch
import torch.nn as nn
from collections import OrderedDict
from models import layer as layer
from models import bricks as bricks
import numpy as np
from models.layer.layer_fundation import Conv2dBatchLeaky as Convolutional


class tiny_yolov3_backbone(nn.Module):
	def __init__(self):
		super().__init__()
		self.input_Channel = 16
		layer_list = [
			# layer 0
			# first layer, smallest_reception
			OrderedDict([
				('0_convbl', layer.Conv2dBatchLeaky(3, 16, 3, 1)),
				('1_maxpool', nn.MaxPool2d(2, 2)),
				('2_convbl', layer.Conv2dBatchLeaky(16, 32, 3, 1)),
				('3_maxpool', nn.MaxPool2d(2, 2)),
				('4_convbl', layer.Conv2dBatchLeaky(32, 64, 3, 1)),
			]),
			# layer 1
			# second layer, larger_reception
			OrderedDict([
				('5_maxpool', nn.MaxPool2d(2, 2)),
				('6_convbl', layer.Conv2dBatchLeaky(64, 128, 3, 1)),

			]),

			#layer 2
			#third layer, more_larger_reception
			OrderedDict([
				('7_maxpool', nn.MaxPool2d(2, 2)),
				('8_convbl', layer.Conv2dBatchLeaky(128, 256, 3, 1)),
			]),
			# layer 3
			# fourth layer, largest_reception
			OrderedDict([
				('9_maxpool', nn.MaxPool2d(2, 2)),
				('10_convbl', layer.Conv2dBatchLeaky(256, 512, 3, 1)),
				('11_maxpool', nn.MaxPool2d(3, 1, padding=1)),
				('12_convbl', layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
				('13_convbl', layer.Conv2dBatchLeaky(1024, 256, 1, 1)),
			]),

			OrderedDict([
				('9_maxpool', nn.MaxPool2d(2, 2)),
				('10_convbl', layer.Conv2dBatchLeaky(256, 512, 3, 1)),
				('11_maxpool', nn.MaxPool2d(3, 1, padding=1)),
				('12_convbl', layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
				('13_convbl', layer.Conv2dBatchLeaky(1024, 256, 1, 1)),
			]),
			OrderedDict([
				('9_maxpool', nn.MaxPool2d(2, 2)),
				('10_convbl', layer.Conv2dBatchLeaky(256, 512, 3, 1)),
				('11_maxpool', nn.MaxPool2d(3, 1, padding=1)),
				('12_convbl', layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
				('13_convbl', layer.Conv2dBatchLeaky(1024, 256, 1, 1)),
			]),
		]
		self.layer = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

	def load_darknet_weights(self, weight_file, cutoff=52):#加载成功
		"https://github.com/ultralytics/yolov3/blob/master/models.py"

		print("load darknet weights : ", weight_file)

		with open(weight_file, 'rb') as f:
			_ = np.fromfile(f, dtype=np.int32, count=5)
			weights = np.fromfile(f, dtype=np.float32)
			print("weights.shape:{}".format(weights.shape))
		count = 0
		ptr = 0
		for m in self.modules():
			if isinstance(m, Convolutional):
				# only initing backbone conv's weights
				if count == cutoff:
					break
				count += 1
				#conv_layer = m._Convolutional__conv
				for sub_m in m.modules():
					if isinstance(sub_m, nn.Conv2d):
						conv_layer = sub_m
					elif isinstance(sub_m, nn.BatchNorm2d):
						bn_layer = sub_m

				# Load BN bias, weights, running mean and running variance
				num_b = bn_layer.bias.numel()  # Number of biases
				# Bias
				bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
				bn_layer.bias.data.copy_(bn_b)
				ptr += num_b
				# Weight
				bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
				bn_layer.weight.data.copy_(bn_w)
				ptr += num_b
				# Running Mean
				bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
				bn_layer.running_mean.data.copy_(bn_rm)
				ptr += num_b
				# Running Var
				bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
				bn_layer.running_var.data.copy_(bn_rv)
				ptr += num_b

				print("loading weight {}".format(bn_layer))
				# else:
				#     # Load conv. bias
				#     num_b = conv_layer.bias.numel()
				#     conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
				#     conv_layer.bias.data.copy_(conv_b)
				#     ptr += num_b
				# Load conv. weights
				num_w = conv_layer.weight.numel()
				conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
				conv_layer.weight.data.copy_(conv_w)
				ptr += num_w
				print("loading weight {}".format(conv_layer))
		print("ptr:{}".format(ptr))
		if ptr == weights.shape[0]:
			print("convert success!")

	def forward(self, input):
		stage4 = self.layer[0](input)
		stage5 = self.layer[1](stage4)
		stage6 = self.layer[2](stage5)
		stage7 = self.layer[3](stage6)

		#test_append
		stage8 = self.layer[4](stage7)
		stage9 = self.layer[5](stage8)

		features = [stage4, stage5, stage6, stage7, stage8, stage9]
		return features

if __name__ == '__main__':
	from tools.cal_reception_filed_tool.cal_RF import calc_receptive_filed
	from tools.cal_effect_field_tool import calculate_EPR
	input = torch.ones(1, 3, 416, 416)
	model = tiny_yolov3_backbone()
	#model.load_darknet_weights('/home/xyl/桌面/yolov3-tiny.weights')
	#model.eval()
	calc_receptive_filed(model, (640, 640, 3), index=[14, 18, 27, 38])
	calculate_EPR(model)
	#
	# input = torch.randn(1, 3, 416, 416)
	# model = tiny_yolov3_backbone()
	# features = model(input)
	# for i in features:
	# 	print(i.shape)

