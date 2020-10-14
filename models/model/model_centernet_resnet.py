import torch.nn as nn
from models.backbone.centernet_resnet18.resnet import resnet18
from models.bricks.centernet_bricks import Conv2d, DeConv2d, SPP
from models.loss.centernet_loss_module import centernet_loss_module
from utils.logger import print_logger
import numpy as np
import torch
from utils.utils_select_device import select_device


class centernet_18(nn.Module):
	def __init__(self, config, logger=None, init_weight=True):
		super().__init__()
		if init_weight:
			self.backbone = resnet18(pretrained=True)
		else:
			self.backbone = resnet18()
		self.loss = centernet_loss_module(config, stride=4)
		if logger == None:
			self.logger = print_logger()
		else:
			self.logger = logger


		self.smooth5 = nn.Sequential(
			SPP(),
			Conv2d(512*4, 256, ksize=1),
			Conv2d(256, 512, ksize=3, padding=1)
		)

		self.deconv5 = DeConv2d(512, 256, ksize=4, stride=2) # 32 -> 16
		self.deconv4 = DeConv2d(256, 256, ksize=4, stride=2) # 16 -> 8
		self.deconv3 = DeConv2d(256, 256, ksize=4, stride=2) #  8 -> 4

		self.num_classes = config['yolo']['classes']

		self.cls_pred = nn.Sequential(
			Conv2d(256, 64, ksize=3, padding=1, leakyReLU=True),
			nn.Conv2d(64, self.num_classes, kernel_size=1)
		)

		self.txty_pred = nn.Sequential(
			Conv2d(256, 64, ksize=3, padding=1, leakyReLU=True),
			nn.Conv2d(64, 2, kernel_size=1)
		)

		self.twth_pred = nn.Sequential(
			Conv2d(256, 64, ksize=3, padding=1, leakyReLU=True),
			nn.Conv2d(64, 2, kernel_size=1)
		)


	def forward(self, input, target=None):
		# backbone
		_, _, _, c5 = self.backbone(input)

		# deconv
		p5 = self.smooth5(c5)
		p4 = self.deconv5(p5)
		p3 = self.deconv4(p4)
		p2 = self.deconv3(p3)

		# head
		cls_pred = self.cls_pred(p2)
		txty_pred = self.txty_pred(p2)
		twth_pred = self.twth_pred(p2)

		center_loss_input = torch.cat((txty_pred, twth_pred, cls_pred), dim=1)

		loss_or_output = self.loss(center_loss_input, target)
		return loss_or_output #input=416,[13, 26, 52]

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

