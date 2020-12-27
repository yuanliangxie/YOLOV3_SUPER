'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version:
@Author: Aoru Xue
@Date: 2019-09-09 23:13:46
@LastEditors: Aoru Xue
@LastEditTime: 2019-10-02 18:39:48
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


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
			nn.ReLU6(channels),
			SeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(channels),
			nn.ReLU6(channels)
		)
		self.relu = nn.ReLU6(channels)

	def forward(self, x):
		return self.relu(x + self.conv2dRelu(x))


class LFFD_LVnet_backbone(nn.Module):
	def __init__(self):
		super(LFFD_LVnet_backbone, self).__init__()
		self.num_classes = 3
		self.priors = None
		self.c1 = nn.Sequential(
			SeparableConv2d(3, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU6(64)
		)
		self.c2 = nn.Sequential(
			SeparableConv2d(64, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU6(64)
		)
		self.tinypart1 = nn.Sequential(
			ResBlock(64),
			ResBlock(64),
			ResBlock(64)
		)
		self.tinypart2 = ResBlock(64)
		self.c11 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU6(64)
		)
		self.smallpart1 = ResBlock(64)
		self.smallpart2 = ResBlock(64)
		self.c16 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU6(128)
		)
		self.mediumpart = ResBlock(128)
		self.c19 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU6(128)
		)
		self.largepart1 = ResBlock(128)


	def forward(self, x):
		c1 = self.c1(x)
		c2 = self.c2(c1)

		c8 = self.tinypart1(c2)  # 4x
		c10 = self.tinypart2(c8)  # 4x

		c11 = self.c11(c10)
		c13 = self.smallpart1(c11)  # 8x 64
		c15 = self.smallpart2(c13)  # 8x 64

		c16 = self.c16(c15)
		c18 = self.mediumpart(c16)  # 16x 128

		c19 = self.c19(c18)
		c21 = self.largepart1(c19)  # 32x 128

		return c13, c15, c18, c21

if __name__ == '__main__':
	from tools.cal_reception_filed_tool.cal_RF import calc_receptive_filed
	from tools.cal_effect_field_tool import calculate_EPR
	#torch.manual_seed(1)
	lffd_backbone = LFFDbackbone()

	calc_receptive_filed(lffd_backbone, (640, 640, 3), index=[34, 43, 46, 76, 88, 97, 106])
#lffd_backbone.eval()
# train_weight = torch.load("/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/size640x640_try_LFFD_test_UA_detrac/20201203144225/model_map_0.910.pth")
# lffd_backbone.load_state_dict(state_dict=train_weight['state_dict'], strict=False)
#calculate_EPR(lffd_backbone)
