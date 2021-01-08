import torch.nn as nn
from models.loss.LVnet_iou_assign_loss.LVnet_iou_assign_centernet_loss_module import centernet_loss_module
from models.backbone.LVnet.LV_net import LV_Net_backbone as backbone
from models.head.LVnet_head import LVnetHead as head
from models.backbone.LVnet.LV_net import Conv2dBatchRelu6
from models.bricks.centernet_bricks import SPP
from models.loss.LVnet_iou_assign_loss.LVnet_iou_assign_loss import LVnetloss_module
from models.assign_target.LVnet_iou_assign_target import assign_targets
from utils.logger import print_logger
import numpy as np
import torch
from utils.utils_select_device import select_device
from tools.time_analyze import func_line_time

class DeConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, ksize, stride=2, leakyReLU=False):
		super(DeConv2d, self).__init__()
		# deconv basic config
		if ksize == 4:
			padding = 1
			output_padding = 0
		elif ksize == 3:
			padding = 1
			output_padding = 1
		elif ksize == 2:
			padding = 0
			output_padding = 0

		self.convs = nn.Sequential(
			nn.ConvTranspose2d(in_channels, out_channels, ksize, stride=stride, padding=padding, output_padding=output_padding),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU6(inplace=True)
		)

	def forward(self, x):
		return self.convs(x)

class LVnet(nn.Module):
	def __init__(self, config, logger=None, init_weight=True):
		super().__init__()
		self.backbone = backbone()
		#self.neck = neck()
		self.head1 = head(in_channels=128, out_channels=64, nClass=config["model"]["classes"])
		self.head2 = head(in_channels=128, out_channels=64, nClass=config["model"]["classes"])
		self.head3 = head(in_channels=256, out_channels=128, nClass=config["model"]["classes"])
		self.head4 = head(in_channels=256, out_channels=256, nClass=config["model"]["classes"])

		self.smooth = nn.Sequential(
			SPP(),
			Conv2dBatchRelu6(1024, 128, kernel_size=1, stride=1, padding=0),
			Conv2dBatchRelu6(128, 256, kernel_size=3, stride=1, padding=1)
		)

		self.deconv4 = DeConv2d(256, 128, ksize=4, stride=2)
		self.deconv3 = DeConv2d(256, 64, ksize=4, stride=2)
		self.deconv2 = DeConv2d(128, 64, ksize=4, stride=2)

		self.centernet_loss_4x = centernet_loss_module(config, stride=4, anchor=[28, 21])
		self.centerent_loss_8x = centernet_loss_module(config, stride=8, anchor=[57, 38])
		self.loss = LVnetloss_module(config)
		self.assign = assign_targets((640, 640), config['device_id'])
		if logger == None:
			self.logger = print_logger()
		else:
			self.logger = logger
		if init_weight:
			self.__init_weights()

	#@func_line_time
	def forward(self, input, target=None):
		features = self.backbone(input)
		#neck_features = self.neck(features)
		f1, f2, f3, f4 = features
		f4 = self.smooth(f4)
		f3 = torch.cat([f3, self.deconv4(f4)], dim=1) #尝试用加法试试
		f2 = torch.cat([f2, self.deconv3(f3)], dim=1)
		f1 = torch.cat([f1, self.deconv2(f2)], dim=1)
		neck_features = [f1, f2, f3, f4]
		head1 = self.head1(neck_features[0])
		head2 = self.head2(neck_features[1])
		head3 = self.head3(neck_features[2])
		head4 = self.head4(neck_features[3])
		LVnet_loss_input= [head1, head2, head3, head4]
		if target != None:
			target_tensor, n_obj= self.assign(target, [input.clone().detach() for input in LVnet_loss_input])
			loss_or_output_1 = self.centernet_loss_4x(LVnet_loss_input[0], target_tensor[0], n_obj)
			loss_or_output_2 = self.centerent_loss_8x(LVnet_loss_input[1], target_tensor[1], n_obj)
			loss_or_output_3_4 = self.loss(LVnet_loss_input[2:], target_tensor[2:], n_obj)
		else:
			loss_or_output_1 = self.centernet_loss_4x(LVnet_loss_input[0], None, None)
			loss_or_output_2 = self.centerent_loss_8x(LVnet_loss_input[1], None, None)
			loss_or_output_3_4 = self.loss(LVnet_loss_input[2:], None, None)
		loss_or_output = loss_or_output_1+loss_or_output_2+loss_or_output_3_4
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

