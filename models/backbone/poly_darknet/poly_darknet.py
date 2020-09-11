import torch
import torch.nn as nn
from collections import OrderedDict
from models import layer as layer
from models import bricks as bricks


class darknet53(nn.Module):
	def __init__(self):
		super().__init__()
		self.input_Channel = 32
		stage_cfg = {'stage_2': 1, 'stage_3': 2, 'stage_4': 8, 'stage_5': 8, 'stage_6': 4}
		layer_list = [
			# layer 0
			# first layer, tiny_reception
			OrderedDict([
				('stage1', layer.Conv2dBatchLeaky(3, 32, 3, 1)),
				('stage2', bricks.CBL_stack(32, stage_cfg['stage_2'])),
				('stage3', bricks.CBL_stack(64, stage_cfg['stage_3'])),
			]),

			# layer 1
			# second layer, small_reception
			OrderedDict([('stage4', bricks.CBL_stack(128, stage_cfg['stage_4']))]),

			# layer 2
			# third layer, middle_reception
			OrderedDict([
				('stage5', bricks.CBL_stack(256, stage_cfg['stage_5']))
			]),

			# layer 3
			# fourth layer, large_reception
			OrderedDict([
				('stage6', bricks.CBL_stack(512, stage_cfg['stage_6']))
			]),
		]
		self.layer = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

	def forward(self, input):
		#这里的tiny-large范围指的是这些特征图的感受野的大小
		tiny = self.layer[0](input)
		small = self.layer[1](tiny)
		middle = self.layer[2](small)
		large = self.layer[3](middle)
		features = [tiny, small, middle, large]
		return features

if __name__ == '__main__':
	#from models.backbone.poly_darknet.poly_neck import neck
	input = torch.randn(1, 3, 416, 416)
	darknet = darknet53()
	#neck = neck()
	features = darknet(input)
	for i in features:
		print(i.shape)
	# features = neck(features)
	# print(features.shape)
