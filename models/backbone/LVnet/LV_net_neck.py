import torch
import torch.nn as nn
from collections import OrderedDict
from models.backbone.LVnet.LV_net import SeparableConv2d, Conv2dBatchRelu6


class stack_to_head(nn.Module):
	def __init__(self, input_channel, first_head=False, last_head=False):
		super().__init__()
		if first_head:
			half_nchannels = int(input_channel / 2)
		elif last_head:
			half_nchannels = 64
		else:
			half_nchannels = int(input_channel / 3)
		in_nchannels = 2 * half_nchannels
		layers = [
			Conv2dBatchRelu6(input_channel, half_nchannels, 1, 1, 0),
			SeparableConv2d(half_nchannels, in_nchannels, 3, 1),
			Conv2dBatchRelu6(in_nchannels, half_nchannels, 1, 1, 0)
		]
		self.feature = nn.Sequential(*layers)

	def forward(self, data):
		x = self.feature(data)
		return x

class Upsample(nn.Module):
	"""
	the module is not pure the literal meaning as upsample,but the module is
	used to upsample function in yolov3, it contains two child-modules:
	conv2dBatchLeaky and Upsample

	"""
	def __init__(self, input_channel):
		super(Upsample, self).__init__()
		self.input_channel = input_channel
		half_nchannels = int(input_channel / 2)
		layers = [
			Conv2dBatchRelu6(self.input_channel, half_nchannels, 1, 1, 0),
			nn.Upsample(scale_factor=2)
		]

		self.features = nn.Sequential(*layers)

	def forward(self, data):
		x = self.features(data)
		return x

class neck(nn.Module):
	def __init__(self):
		super().__init__()
		layer_list = [
			# largePart
			# output the layer recep_largest
			OrderedDict([
				('stack_to_head1',  stack_to_head(256, first_head=True))
			]),

			# middlePart
			# output the layer recep middle

			OrderedDict([
				('stack_to_head2', stack_to_head(192, first_head=False))
			]),

			# smallPart
			# output the layer recep small

			OrderedDict([
				('stack_to_head3', stack_to_head(96, first_head=False, last_head=True))
			]),

			#tinyPart
			OrderedDict([
				('stack_to_head4', stack_to_head(96, first_head=False, last_head=True))
			]),

			#largepart
			# UpSample, connect the reception_largest  to reception_second

			OrderedDict([
				('upsample1', Upsample(128))
			]),

			# middlePart
			# UpSample, connect the reception_middle to reception_small
			OrderedDict([
				('upsample2', Upsample(64))
			]),

			#smallPart
			# UpSample, connect the reception_middle to reception_small
			OrderedDict([
				('upsample3', Upsample(64))
			])

		]
		self.layer = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

	def forward(self, input):
		tinyPart, smallPart, middlePart, largePart = input


		stack_to_head1 = self.layer[0](largePart)

		upsample1 = self.layer[4](stack_to_head1)
		concat1 = torch.cat([upsample1, middlePart], 1)

		# second output the middle reception
		stack_to_head2 = self.layer[1](concat1)

		upsample2 = self.layer[5](stack_to_head2)
		concat2 = torch.cat([upsample2, smallPart], 1)

		# third output the smallest rception
		stack_to_head3 = self.layer[2](concat2)

		upsample3 = self.layer[6](stack_to_head3)

		concat3 = torch.cat([upsample3, tinyPart], 1)

		#fourth
		stack_to_head4 = self.layer[3](concat3)

		features = [stack_to_head4, stack_to_head3, stack_to_head2, stack_to_head1]

		return features


###########################neck_false#######################################
class neck_false(nn.Module):
	def __init__(self):
		super().__init__()
		layer_list = [
			# largePart
			# output the layer recep_largest
			OrderedDict([
				('stack_to_head1',  stack_to_head(256, first_head=True))
			]),

			# middlePart
			# output the layer recep middle

			OrderedDict([
				('stack_to_head2', stack_to_head(128, first_head=False, last_head=True))
			]),

			# smallPart
			# output the layer recep small

			OrderedDict([
				('stack_to_head3', stack_to_head(64, first_head=False, last_head=True))
			]),

			#tinyPart
			OrderedDict([
				('stack_to_head4', stack_to_head(64, first_head=False, last_head=True))
			]),



		]
		self.layer = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

	def forward(self, input):
		tinyPart, smallPart, middlePart, largePart = input


		stack_to_head1 = self.layer[0](largePart)


		# second output the middle reception
		stack_to_head2 = self.layer[1](middlePart)


		# third output the smallest rception
		stack_to_head3 = self.layer[2](smallPart)


		#fourth
		stack_to_head4 = self.layer[3](tinyPart)

		features = [stack_to_head4, stack_to_head3, stack_to_head2, stack_to_head1]

		return features

class LV_backbone(nn.Module):
	def __init__(self):
		super().__init__()
		self.backbone = LV_Net_backbone()
		self.neck = neck()
	def forward(self, x):
		x = self.backbone(x)
		output = self.neck(x)
		return output
if __name__ == '__main__':
	from tools.cal_reception_filed_tool.cal_RF import calc_receptive_filed
	from models.backbone.LVnet.LV_net import LV_Net_backbone
	from tools.cal_effect_field_tool import calculate_EPR
	input = torch.ones(1, 3, 640, 640)

	backbone = LV_Net_backbone()
	#neck1 = neck()

	features = backbone(input)
	#outputs = neck1(features)

	for output in features:
		print(output.shape)

	# backbone = LV_backbone()
	# calculate_EPR(backbone)
	#calc_receptive_filed(backbone,(640, 640, 3), index=[110, 122, 134, 146])