import torchvision
import torch.nn as nn
import torch

class VGG(nn.Module):

	def __init__(self, features, num_classes=1000, init_weights=True):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		# self.classifier = nn.Sequential(
		# 	nn.Linear(512 * 7 * 7, 4096),
		# 	nn.ReLU(True),
		# 	nn.Dropout(),
		# 	nn.Linear(4096, 4096),
		# 	nn.ReLU(True),
		# 	nn.Dropout(),
		# 	nn.Linear(4096, num_classes),
		# )
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x1 = self.features(x)
		#x2 = self.avgpool(x1)
		# x = torch.flatten(x, 1)
		# x = self.classifier(x)
		return x1

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


cfgs = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

if __name__ == '__main__':
	from tools.cal_reception_filed_tool.cal_RF import calc_receptive_filed
	from tools.cal_effect_field_tool import calculate_EPR
	vgg16 = VGG(make_layers(cfgs['D'], batch_norm=False))
	calc_receptive_filed(vgg16, (640, 640, 3), index=[-1])
	calculate_EPR(vgg16)
