#from models.loss.poly_yolo_loss import YOLOLoss as YOLOLoss
from models.loss.poly_yolo_loss_modify import YOLOLoss
import torch.nn as nn
class yolo_loss_module(nn.Module):
	def __init__(self, config, stride=4):
		super().__init__()
		self.yolo_loss = YOLOLoss(config["yolo"]["classes"], stride, config=config, device_id= config["device_id"])

	def forward(self, input, target=None):
		result = []
		result.append(self.yolo_loss(input, target))
		return result
