import torch
import torch.nn as nn
from collections import OrderedDict
from models import layer as layer


class tiny_neck(nn.Module):
	def __init__(self):
		super().__init__()
		layer_list = [

			OrderedDict([
				('stack_to_head1', layer.Conv2dBatchLeaky(256, 128, 1, 1))
			]),


			OrderedDict([
				('upsample1', nn.Upsample(scale_factor=2))
			]),
		]
		self.layer = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

	def forward(self, input):

		stage_26, stage_13 = input
		stack_to_head1 = self.layer[0](stage_13)
		upsample1 = self.layer[1](stack_to_head1)
		concat1 = torch.cat([upsample1, stage_26], 1)
		return [concat1, stage_13]
if __name__ == '__main__':
	from models.backbone.tiny_yolov3.tiny_yolov3_backbone import tiny_yolov3_backbone
	backbone = tiny_yolov3_backbone()
	tiny_neck = neck()
	input = torch.randn(1, 3, 416, 416)
	temp = backbone(input)
	output = tiny_neck(temp)
	for i in (output):
		print(i.shape)

