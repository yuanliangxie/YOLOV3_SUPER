import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBatchRelu6(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride= 1, padding=1):
		super(Conv2dBatchRelu6, self).__init__()
		self.layer = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
			nn.BatchNorm2d(out_channels),
			nn.ReLU6()
		)
	def forward(self, input):
		return self.layer(input)

class SeparableConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
		super(SeparableConv2d, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels,
					  padding=padding),
			nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1))
		)

	def forward(self, x):
		return self.conv(x)


class ResBlock(nn.Module):
	def __init__(self, channels):
		super(ResBlock, self).__init__()
		self.conv2dRelu = nn.Sequential(
			SeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(channels),
			nn.ReLU6(),
			SeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(channels),
			nn.ReLU6()
		)
		self.relu = nn.ReLU6()

	def forward(self, x):
		return self.relu(x + self.conv2dRelu(x))


class LV_Net_backbone(nn.Module):
	def __init__(self):
		super(LV_Net_backbone, self).__init__()
		self.c1 = nn.Sequential(
			SeparableConv2d(3, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU6(),
			#nn.MaxPool2d(2, 2),
		)
		self.c2 = nn.Sequential(
			SeparableConv2d(64, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU6(),
			#nn.MaxPool2d(2, 2),
		)
		self.tinypart1 = nn.Sequential(
			ResBlock(64),
			ResBlock(64),
			ResBlock(64),
			ResBlock(64),
			ResBlock(64),
			ResBlock(64),
		)
		self.c15 = nn.Sequential(
			# SeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1),
			Conv2dBatchRelu6(64, 64, kernel_size=3, stride=2, padding=1),
			#nn.MaxPool2d(2, 2),
		)
		self.smallpart1 = nn.Sequential(
			ResBlock(64),
			#Conv2dBatchLeaky(64, 64, kernel_size=1, stride=1, padding=0)
		)
		self.c18 = nn.Sequential(
			Conv2dBatchRelu6(64, 128, kernel_size=3, stride=2, padding=1),
			#nn.MaxPool2d(2, 2)
		)
		self.mediumpart = nn.Sequential(
			ResBlock(128),
			#Conv2dBatchRelu6(128, 128, kernel_size=3, stride=1, padding=1)
		)
		self.c21 = nn.Sequential(
			Conv2dBatchRelu6(128, 256, kernel_size=3, stride=2, padding=1),
			#nn.MaxPool2d(2, 2)
		)
		self.largepart = ResBlock(256)

	def forward(self, x):
		c1 = self.c1(x)
		c2 = self.c2(c1)

		c14 = self.tinypart1(c2)  # 4x

		c15 = self.c15(c14)
		c17 = self.smallpart1(c15) # 8x

		c18 = self.c18(c17)
		c20 = self.mediumpart(c18)  # 16x

		c21 = self.c21(c20)
		c23 = self.largepart(c21)  # 32x

		return c14, c17, c20, c23

if __name__ == '__main__':
	from tools.cal_reception_filed_tool.cal_RF import calc_receptive_filed
	from tools.cal_effect_field_tool import calculate_EPR
	#torch.manual_seed(1)
	lffd_backbone = LV_Net_backbone()
	input = torch.ones((1, 3, 640, 640))
	fps = lffd_backbone(input)
	for fp in fps:
		print(fp.shape)

	calc_receptive_filed(lffd_backbone, (640, 640, 3), index=[3, 7, 61, 64, 73, 76, 85, 88, 97])
	#lffd_backbone.eval()
	# train_weight = torch.load("/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/size640x640_try_LFFD_test_UA_detrac/20201203144225/model_map_0.910.pth")
	# lffd_backbone.load_state_dict(state_dict=train_weight['state_dict'], strict=False)
	#calculate_EPR(lffd_backbone)
