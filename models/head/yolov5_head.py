import torch.nn as nn
from collections import OrderedDict
from .. layer import layer_fundation as layer

class yolov5_head(nn.Module):
	def __init__(self, nAnchors, nClass, gw=0.5):
		super().__init__()
		big_ = round(256 * gw)
		middle = round(512 * gw)
		small_ = round(1024 * gw)
		self.out_big = nn.Sequential(
			nn.Conv2d(big_, nAnchors * (5 + nClass), 1, 1, 0)
		)
		self.out_middle = nn.Sequential(
			nn.Conv2d(middle, nAnchors * (5 + nClass), 1, 1, 0)
		)
		self.out_small = nn.Sequential(
			nn.Conv2d(small_, nAnchors * (5 + nClass), 1, 1, 0)
		)

	def forward(self, features):
		out_set_tie_big, up_middle, out_set_small = features
		out_small = self.out_small(out_set_small)
		out_middle = self.out_middle(up_middle)
		out_big = self.out_big(out_set_tie_big)
		return out_small, out_middle, out_big
