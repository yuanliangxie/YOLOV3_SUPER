import torch.nn as nn
import torch
from collections import OrderedDict
from .. layer import layer_fundation as layer

class LVnetHead(nn.Module):
	def __init__(self, in_channels, out_channels=64, nClass=1):
		super().__init__()
		self.conv1x1relu = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(out_channels),
			nn.ReLU6()
		)

		#这里结合一起推理好呢？还是分开来进行推理好？
		self.score = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(out_channels),
			nn.ReLU6(),
			nn.Conv2d(out_channels, nClass, kernel_size=1, stride=1)
		)

		self.locations = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(out_channels),
			nn.ReLU6(),
			nn.Conv2d(out_channels, 4, kernel_size=1, stride=1)
		)

	def forward(self, feature):
		score = self.score(self.conv1x1relu(feature))
		location = self.locations(self.conv1x1relu(feature))
		output = torch.cat([location, score], 1)
		return output
