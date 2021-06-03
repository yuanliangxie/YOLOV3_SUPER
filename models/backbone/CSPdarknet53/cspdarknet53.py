import torch
import torch.nn as nn

class SiLU(torch.nn.Module):  # export-friendly version of nn.SiLU()
	@staticmethod
	def forward(x):
		return x * torch.sigmoid(x)


def autopad(k, p=None):  # kernel, padding
	# Pad to 'same'
	if p is None:
		p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
	return p


class CBL(nn.Module):

	def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, e=1.0):
		super(CBL, self).__init__()
		c1 = round(c1 * e)
		c2 = round(c2 * e)
		self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
		self.bn = nn.BatchNorm2d(c2)
		self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

	def forward(self, x):
		return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):

	def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True, e=1.0):
		super(Focus, self).__init__()
		c2 = round(c2 * e)
		self.conv = CBL(c1 * 4, c2, k, s, p, g, act)

	def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
		flatten_channel = torch.cat([x[..., 0::2, 0::2],
									 x[..., 1::2, 0::2],
									 x[..., 0::2, 1::2],
									 x[..., 1::2, 1::2]], dim=1)
		return self.conv(flatten_channel)


class SPP(nn.Module):

	def __init__(self, c1, c2, k=(5, 9, 13), e=1.0):
		super(SPP, self).__init__()
		c1 = round(c1 * e)
		c2 = round(c2 * e)
		c_ = c1 // 2
		self.cbl_before = CBL(c1, c_, 1, 1)
		self.max_pool = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
		self.cbl_after = CBL(c_ * 4, c2, 1, 1)

	def forward(self, x):
		x = self.cbl_before(x)
		x_cat = torch.cat([x] + [m(x) for m in self.max_pool], 1)
		return self.cbl_after(x_cat)


class ResUnit_n(nn.Module):

	def __init__(self, c1, c2, n):
		super(ResUnit_n, self).__init__()
		self.shortcut = c1 == c2
		res_unit = nn.Sequential(
			CBL(c1, c1, k=1, s=1, p=0),
			CBL(c1, c2, k=3, s=1, p=1)
		)
		self.res_unit_n = nn.Sequential(*[res_unit for _ in range(n)])

	def forward(self, x):
		return x + self.res_unit_n(x) if self.shortcut else self.res_unit_n(x)


class CSP1_n(nn.Module):

	def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, n=1, e=None):
		super(CSP1_n, self).__init__()

		c1 = round(c1 * e[1])
		c2 = round(c2 * e[1])
		n = round(n * e[0])
		c_ = c2 // 2
		self.up = nn.Sequential(
			CBL(c1, c_, k, s, autopad(k, p), g, act),
			ResUnit_n(c_, c_, n),
			# nn.Conv2d(c_, c_, 1, 1, 0, bias=False) 这里最新yolov5结构中去掉了，与网上的结构图稍微有些区别
		)
		self.bottom = nn.Conv2d(c1, c_, 1, 1, 0)
		self.tie = nn.Sequential(
			nn.BatchNorm2d(c_ * 2),
			nn.LeakyReLU(),
			nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)
		)
	def forward(self, x):
		total = torch.cat([self.up(x), self.bottom(x)], dim=1)
		out = self.tie(total)
		return out

class CSPDarkNet(nn.Module):

	def __init__(self, gd=0.33, gw=0.5):
		super(CSPDarkNet, self).__init__()
		self.truck_big = nn.Sequential(
			Focus(3, 64, e=gw),
			CBL(64, 128, k=3, s=2, p=1, e=gw),
			CSP1_n(128, 128, n=3, e=[gd, gw]),
			CBL(128, 256, k=3, s=2, p=1, e=gw),
			CSP1_n(256, 256, n=9, e=[gd, gw]),

		)
		self.truck_middle = nn.Sequential(
			CBL(256, 512, k=3, s=2, p=1, e=gw),
			CSP1_n(512, 512, n=9, e=[gd, gw]),
		)
		self.truck_small = nn.Sequential(
			CBL(512, 1024, k=3, s=2, p=1, e=gw),
			SPP(1024, 1024, e=gw)
		)

	def forward(self, x):
		h_big = self.truck_big(x)  # torch.Size([2, 128, 76, 76])
		h_middle = self.truck_middle(h_big)
		h_small = self.truck_small(h_middle)
		features = [h_big, h_middle, h_small]
		return features
if __name__ == '__main__':
	test_image = torch.ones(1, 3, 416, 416)
	model = CSPDarkNet()
	features = model(test_image)
	for feature in features:
		print(feature.shape)