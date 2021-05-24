from models.loss.tiny_yolo_loss.tiny_yolov3_loss import YOLOLoss as YOLOLoss
import torch.nn as nn
class yolo_loss_module(nn.Module):
	def __init__(self, config, strides=[16, 32]):
		super().__init__()
		self.strides = strides
		yolo_losses = []
		for i in range(len(strides)):
			yolo_losses.append(YOLOLoss(config["model"]["anchors"][i],
										config["model"]["classes"], strides[i], config=config, device_id = config["device_id"]))
		self.layers = nn.ModuleList([yolo_losses[i] for i in range(len(strides))])

	def forward(self, input, target=None):
		result = []
		for i in range(len(self.strides)):
			loss_or_output_data = self.layers[i](input[i], target)
			result.append(loss_or_output_data)
		return result
