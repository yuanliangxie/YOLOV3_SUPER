import torch.nn as nn
from models.backbone.LFFD.lffdnet import LFFDbackbone
from models.loss.LFFD_loss_module import LFFD_loss_module
from utils.logger import print_logger
import torch
import math


class LFFDLossBranch(nn.Module):
	def __init__(self, in_channels, out_channels=64, num_classes=3):
		super(LFFDLossBranch, self).__init__()
		self.conv1x1relu = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(out_channels),
			nn.ReLU6(out_channels)
		)

		#这里结合一起推理好呢？还是分开来进行推理好？
		self.score = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(out_channels),
			nn.ReLU6(out_channels),
			nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1)
		)

		self.locations = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(out_channels),
			nn.ReLU6(out_channels),
			nn.Conv2d(out_channels, 4, kernel_size=1, stride=1)
		)

	def forward(self, x):
		score = self.score(self.conv1x1relu(x))
		locations = self.locations(self.conv1x1relu(x))
		output_featrue_map = torch.cat([score, locations], dim=1)
		return output_featrue_map


class assign_targets(object):
	def __init__(self, input_shape):
		"""
		:param target: b,t,5  [cls, x, y, w, h]
		:param input_shape: (h,w)
		"""
		self.strides = [4, 4, 8, 8, 16, 32, 32, 32]
		self.in_h = [input_shape[0]/s for s in self.strides]
		self.in_w = [input_shape[1]/s for s in self.strides]
		self.continuous_face_scale=[[10,15], [15, 20], [20, 40], [40, 70], [70, 110], [110, 250], [250, 400], [400, 560]]


	def __call__(self, target):
		if target == None:
			return None, 0

		self.target = target
		bs = self.target.shape[0]
		ts = self.target.shape[1]
		self.t_scales = []
		for i, strides in enumerate(self.strides):
			self.t_scale = torch.zeros((bs, int(self.in_h[i]), int(self.in_w[i]), 6), requires_grad=False)
			self.t_scales.append(self.t_scale)
		n_obj = 0
		for b in range(bs):
			for t in range(ts):
				if self.target[b, t].sum() == 0:
					continue
				else:
					n_obj = n_obj + 1
					x = [self.target[b, t, 1] * in_w for in_w in self.in_w]
					y = [self.target[b, t, 2] * in_h for in_h in self.in_h]
					w = [self.target[b, t, 3] * in_w for in_w in self.in_w]
					h = [self.target[b, t, 4] * in_h for in_h in self.in_h]

					scale_xywh = list(zip(x, y, w, h))
					max_border = [max(couple[0], couple[1]) for couple in zip(w, h)]
					for i,  scale in enumerate(self.continuous_face_scale):
						if scale[0]/self.strides[i] <= max_border[i] <= scale[1]/self.strides[i]:
							self.assign_scale(self.t_scales[i], scale_xywh[i], b, t, scale_index=i) #为某层赋予真实标签！
							continue
						elif i== (len(self.continuous_face_scale)-1) and max_border[i] > scale[1]/self.strides[i]:
							self.assign_scale(self.t_scales[i], scale_xywh[i], b, t, scale_index=i) #如果annotation_area get more than biggets scale
						# else:
						# 	print("真实标注框:" + str(scale_xywh[i]) + "没有被正确匹配到！")

		return self.t_scales, n_obj

	def assign_scale(self, t_scale, scale_xywh, b, t, scale_index):
		gx, gy, gw, gh = scale_xywh

		# Get grid box indices
		gi = int(gx)
		gj = int(gy)

		t_scale[b, gj, gi, 0] = gx - gi
		t_scale[b, gj, gi, 1] = gy - gj
		t_scale[b, gj, gi, 2] = math.log(gw / (sum(self.continuous_face_scale[scale_index])/(2*self.strides[scale_index])) + 1e-16)
		t_scale[b, gj, gi, 3] = math.log(gh / (sum(self.continuous_face_scale[scale_index])/(2*self.strides[scale_index])) + 1e-16)

		t_scale[b, gj, gi, 4] = 1 #cls

		t_scale[b, gj, gi, 5] = 2 - self.target[b, t, 3] * self.target[b, t, 4] #scale_mask

		return 0



class LFFD(nn.Module):
	def __init__(self, config, logger=None, init_weight=True):
		super().__init__()

		self.backbone = LFFDbackbone()
		self.loss = LFFD_loss_module(config)

		if logger == None:
			self.logger = print_logger()
		else:
			self.logger = logger
		self.num_classes = config['model']['classes']
		self.lossbranch1 = LFFDLossBranch(64, num_classes=self.num_classes)
		self.lossbranch2 = LFFDLossBranch(64, num_classes=self.num_classes)
		self.lossbranch3 = LFFDLossBranch(64, num_classes=self.num_classes)
		self.lossbranch4 = LFFDLossBranch(64, num_classes=self.num_classes)
		self.lossbranch5 = LFFDLossBranch(128, num_classes=self.num_classes)
		self.lossbranch6 = LFFDLossBranch(128, num_classes=self.num_classes)
		self.lossbranch7 = LFFDLossBranch(128, num_classes=self.num_classes)
		self.lossbranch8 = LFFDLossBranch(128, num_classes=self.num_classes)

		if init_weight:
			self.__init_weights()

	def __init_weights(self):

		" Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
		for m in self.modules():#
			if isinstance(m, nn.Conv2d):
				torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
				# torch.nn.init.constant_(m.weight.data,0.001)#在测试时为了看模型有没有弄错，进行的改动
				if m.bias is not None:
					m.bias.data.zero_()
				print("initing {}".format(m))

			elif isinstance(m, nn.BatchNorm2d):
				torch.nn.init.constant_(m.weight.data, 1.0)
				torch.nn.init.constant_(m.bias.data, 0.0)

				print("initing {}".format(m))

	def forward(self, x, target=None):
		self.assign = assign_targets(x.shape[2:])
		c8, c10, c13, c15, c18, c21, c23, c25 = self.backbone(x)
		c8 = self.lossbranch1(c8)
		c10 = self.lossbranch2(c10)
		c13 = self.lossbranch3(c13)
		c15 = self.lossbranch4(c15)
		c18 = self.lossbranch5(c18)
		# print(loc1.size(),loc2.size(),loc3.size(),loc4.size(),loc5.size())
		c21 = self.lossbranch6(c21)
		c23 = self.lossbranch7(c23)
		c25 = self.lossbranch8(c25)

		target_tensor, n_obj = self.assign(target)

		loss_or_output = self.loss([c8, c10, c13, c15, c18, c21, c23, c25], target_tensor, n_obj)

		#loss_or_output = self.loss([c8, c10], target_tensor, n_obj) #test1

		#loss_or_output = self.loss([c13, c15, c18, c21], target_tensor, n_obj) #test2

		#loss_or_output = self.loss([c23, c25], target_tensor, n_obj) #test3

		return loss_or_output #input=416,[13, 26, 52]
if __name__ == '__main__':
	from tools.cal_effect_field_tool import calculate_EPR
	from tools.cal_reception_filed_tool.cal_RF import calc_receptive_filed
	config={"device_id":'cpu', "num_classes":1, "model":{"classes":1}}
	model = LFFD(config)
	train_weight = torch.load("/home/xyl/PycharmProjects/YOLOV3_SUPER/darknet53/size640x640_try_LFFD_test_UA_detrac/20201203144225/model_map_0.910.pth")
	print(model.load_state_dict(state_dict=train_weight['state_dict'], strict=True))
	#summary(model,(3,640,640),device = "cpu")
	#calc_receptive_filed(model, (640, 640, 3), index=[i for i in range(101)])
	calculate_EPR(model.backbone)
