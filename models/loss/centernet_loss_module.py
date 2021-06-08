from models.loss.centernet_loss import centernet_Loss
import torch.nn as nn
class centernet_loss_module(nn.Module):
	def __init__(self, config, stride=4):
		super().__init__()
		self.center_loss = centernet_Loss(config["model"]["classes"], stride, config=config, device_id= config["device_id"])

	def forward(self, input, target=None):
		result = []
		result.append(self.center_loss(input, target))
		return result
