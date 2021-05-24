import torch.nn as nn
from collections import OrderedDict
from models import layer as layer

class yolov3_head(nn.Module):
	def __init__(self, nAnchors, nClass):
		super().__init__()
		self.base = 6
		num_filters = self.base * 32
		layer_list = [

			OrderedDict([
			('feature0', layer.Conv2dBatchLeaky(768, num_filters, 1, 1)),
			('feature1', layer.Conv2dBatchLeaky(num_filters, num_filters*2, 3, 1)),
			('feature2', layer.Conv2dBatchLeaky(num_filters*2, num_filters, 1, 1))
				]),

			OrderedDict([
			('feature0', layer.Conv2dBatchLeaky(num_filters, num_filters*2, 3, 1)),
			('feature1', layer.Conv2dBatchLeaky(num_filters*2, nAnchors * (nClass + 5), 1, 1)),
			])
		]


		self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])



	"""
		num_filters = base*32
		
		x = compose(
			DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
			DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
			DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(all)
		
		all = compose(
			DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
			DarknetConv2D(num_anchors * (num_classes + 5 + NUM_ANGLES3), (1, 1)))(x)
	"""

	def forward(self, features):
		features = self.layers[0](features)
		features = self.layers[1](features)
		return features
if __name__ == '__main__':
	import torch
	input = torch.randn(1, 768, 104, 104)
	yolo_head = yolov3_head(1, 5)
	output = yolo_head(input)
	print(output.shape)
