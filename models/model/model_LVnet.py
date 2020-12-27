import torch.nn as nn
from models.backbone.LVnet.LV_net import LV_Net_backbone as backbone
#from models.backbone.LFFD.lffdnet_test import LFFD_LVnet_backbone as backbone
from models.backbone.LVnet.LV_net_neck import neck
from models.head.LVnet_head import LVnetHead as head
from models.loss.LVnet_loss_module_2 import LVnetloss_module as loss
from utils.logger import print_logger
import numpy as np
import torch
from utils.utils_select_device import select_device
import math

class assign_targets(object):
	def __init__(self, input_shape):
		"""
		:param target: b,t,5  [cls, x, y, w, h]
		:param input_shape: (h,w)
		"""
		self.strides = [4, 8, 16, 32]
		self.in_h = [input_shape[0]/s for s in self.strides]
		self.in_w = [input_shape[1]/s for s in self.strides]
		self.continuous_face_scale=[[15, 45], [45, 75], [75, 135], [135, 260]]
		self.anchors = [[28, 21], [57, 38], [100, 65], [164, 121]] #(w, h)


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
					area = [couple[0] *couple[1] for couple in zip(w, h)]
					for i,  scale in enumerate(self.continuous_face_scale):
						if scale[0]*scale[0]/(self.strides[i]**2) <= area[i] <= scale[1]*scale[1]/(self.strides[i]**2):
							self.assign_scale(self.t_scales[i], scale_xywh[i], b, t, scale_index=i) #为某层赋予真实标签！
							continue
						elif i== (len(self.continuous_face_scale)-1) and area[i] > scale[1]*scale[1]/(self.strides[i]**2):
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
		t_scale[b, gj, gi, 2] = math.log(gw / (self.anchors[scale_index][0]/(self.strides[scale_index])) + 1e-16)
		t_scale[b, gj, gi, 3] = math.log(gh / (self.anchors[scale_index][1]/(self.strides[scale_index])) + 1e-16)

		t_scale[b, gj, gi, 4] = 1 #cls

		t_scale[b, gj, gi, 5] = 2 - self.target[b, t, 3] * self.target[b, t, 4] #scale_mask

		return 0
class LVnet(nn.Module):
	def __init__(self, config, logger=None, init_weight=True):
		super().__init__()
		self.backbone = backbone()
		#self.neck = neck()
		self.head1 = head(in_channels=64, out_channels=64, nClass=config["model"]["classes"])
		self.head2 = head(in_channels=64, out_channels=64, nClass=config["model"]["classes"])
		self.head3 = head(in_channels=128, out_channels=128, nClass=config["model"]["classes"])
		self.head4 = head(in_channels=256, out_channels=256, nClass=config["model"]["classes"])

		self.loss = loss(config)
		if logger == None:
			self.logger = print_logger()
		else:
			self.logger = logger
		if init_weight:
			self.__init_weights()

	def forward(self, input, target=None):
		self.assign = assign_targets(input.shape[2:])
		features = self.backbone(input)
		#neck_features = self.neck(features)
		neck_features = features
		head1 = self.head1(neck_features[0])
		head2 = self.head2(neck_features[1])
		head3 = self.head3(neck_features[2])
		head4 = self.head4(neck_features[3])
		LVnet_loss_input= [head1, head2, head3, head4]
		target_tensor, n_obj = self.assign(target)
		loss_or_output = self.loss(LVnet_loss_input, target_tensor, n_obj)
		return loss_or_output #input=416,[13, 26, 52]

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


if __name__ == '__main__':
	import train.Voc_data_preprocess.params_init_voc as params_init
	config = params_init.TRAINING_PARAMS
	device = select_device(0)
	torch.cuda.manual_seed_all(1)
	torch.backends.cudnn.deterministic = True
	torch.manual_seed(1)
	np.random.seed(1)
	net = yolov3(config)
	net.to(device)
	# net.cpu()
	# net.count_darknet_count()
	# for idx, m  in enumerate(net.backbone.layer[0].modules()):
	#     print(idx, "->", m)
	# net.backbone.layer[0].parameters()
	#pretrain_coco_weight_darknet = torch.load("darknet53_weights_pytorch.pth")
	net.load_darknet_weights('../../weights/darknet53.conv.74')
	#net.load_darknet_pth_weights(pth_file = "../../weights/darknet53_weights_pytorch.pth")
	net.eval()
	images = torch.ones((1,3,416,416)).to(device)
	yolo_loss_input = net(images)
	print(yolo_loss_input[0].shape)
	print(yolo_loss_input[0])
	"""
	output:
	tensor([ 1.1618e-05, -2.5806e-04, -1.8426e-04, -1.0144e-06, -8.8483e-05,
				   -2.9103e-05, -4.6486e-05, -5.9855e-05, -3.9318e-05, -4.0554e-05,
				   -6.2083e-05,  2.8495e-05, -2.7813e-04], grad_fn=<SliceBackward>)
	"""

