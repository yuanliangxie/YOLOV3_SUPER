import torch
import torch.nn as nn
from models import layer as layer


class neck(nn.Module):
	def __init__(self):
		super().__init__()
		self.base = 6
		self.layer_tiny = layer.Conv2dBatchLeaky(128, self.base * 32, 1, 1)
		self.layer_small = layer.Conv2dBatchLeaky(256, self.base * 32, 1, 1)
		self.layer_middle = layer.Conv2dBatchLeaky(512, self.base * 32, 1, 1)
		self.layer_large = layer.Conv2dBatchLeaky(1024, self.base * 32, 1, 1)

		self.upsample_1 = nn.Upsample(scale_factor=2)
		self.upsample_2 = nn.Upsample(scale_factor=2)
		self.upsample_3 = nn.Upsample(scale_factor=2)

	def forward(self, input):
		tiny, small, middle, large = input
		tiny = self.layer_tiny(tiny)
		small = self.layer_small(small)
		middle = self.layer_middle(middle)
		large = self.layer_large(large)

		output = torch.cat([middle, self.upsample_1(large)], 1)
		output = torch.cat([small, self.upsample_2(output)], 1)
		output = torch.cat([tiny, self.upsample_3(output)], 1)

		# output = middle + self.upsample_1(large)
		# output = small + self.upsample_2(output)
		# output = tiny + self.upsample_3(output)

		return output
