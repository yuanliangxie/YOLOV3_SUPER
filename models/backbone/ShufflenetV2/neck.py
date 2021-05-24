import torch
import torch.nn as nn
from collections import OrderedDict
from models import layer as layer
from models import bricks as bricks

class stack_to_head(nn.Module):
	def __init__(self, input_channel, first_head=False):
		super().__init__()
		if first_head:
			half_nchannels = int(input_channel / 2)
		else:
			half_nchannels = int(input_channel / 3)
		in_nchannels = 2 * half_nchannels
		layers = [
			layer.Conv2dBatchLeaky(input_channel, half_nchannels, 1, 1),
			layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
			layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1),
			# layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
			# layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
		]
		self.feature = nn.Sequential(*layers)

	def forward(self, data):
		x = self.feature(data)
		return x

class neck(nn.Module):
	def __init__(self):
		super().__init__()
		layer_list = [
			# layer3 - > 0
			# output the layer recep_largest
			OrderedDict([
				('stack_to_head1', stack_to_head(1024, first_head=True))
			]),

			# layer4 -> 1
			# output the layer recep middle

			OrderedDict([
				('stack_to_head2', stack_to_head(352, first_head=False))
			]),

			# layer5 -> 2
			# output the layer recep small

			OrderedDict([
				('stack_to_head3', stack_to_head(106, first_head=False))
			]),

			# layer6 -> 3
			# UpSample, connect the reception_largest  to reception_second

			OrderedDict([
				('upsample1', bricks.Upsample(512))
			]),

			# layer7 -> 4
			# UpSample, connect the reception_middle to reception_small
			OrderedDict([
				('upsample2', bricks.Upsample(117))
			])

		]
		self.layer = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

	def forward(self, input):
		stage4, stage5, stage6 = input
		stack_to_head1 = self.layer[0](stage6)

		upsample1 = self.layer[3](stack_to_head1)
		concat1 = torch.cat([upsample1, stage5], 1)

		# second output the middle reception
		stack_to_head2 = self.layer[1](concat1)

		upsample2 = self.layer[4](stack_to_head2)
		concat2 = torch.cat([upsample2, stage4], 1)
		# third output the smallest rception
		stack_to_head3 = self.layer[2](concat2)
		features = [stack_to_head1, stack_to_head2, stack_to_head3]#input=416,[13, 26, 52]

		return features
