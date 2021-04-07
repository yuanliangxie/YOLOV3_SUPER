import torch.nn as nn
from models.backbone.ShufflenetV2.shufflenetv2 import shufflenet2
from models.backbone.ShufflenetV2.neck import neck
class shufflenetv2_with_fpn(nn.Module):
	def __init__(self):
		super(shufflenetv2_with_fpn, self).__init__()
		self.shufflenetv2 = shufflenet2()
		self.neck = neck()

	def forward(self, input):
		outs = self.shufflenetv2(input)
		outs = self.neck(outs)
		return outs

if __name__ == '__main__':
	import torch
	shufflenet2_model = shufflenetv2_with_fpn()
	input = torch.ones((1, 3, 416, 416))
	outs = shufflenet2_model(input)
	for out in outs:
		print(out.shape)


