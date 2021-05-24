from models.loss.poly_yolo_loss import YOLOLoss as YOLOLoss
import torch.nn as nn
class yolo_loss_module(nn.Module):
	def __init__(self, config, stride=4):
		super().__init__()
		self.yolo_loss = YOLOLoss(config["yolo"]["anchors"],
								  config["yolo"]["classes"], stride , config_anchor=config["yolo"]["anchors"], device_id= config["device_id"])

	def forward(self, input, target=None):
		result = []
		result.append(self.yolo_loss(input, target))
		return result
