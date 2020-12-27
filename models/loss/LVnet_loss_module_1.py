from models.loss.LVnet_loss import LVnetLoss as LVnetLoss
import torch.nn as nn
class LVnetloss_module(nn.Module):
	def __init__(self, config, strides=[4, 8, 16, 32]):
		super().__init__()
		LVnet_losses = []
		for i in range(len(strides)):
			LVnet_losses.append(LVnetLoss(config["model"]["anchors"][i],
										config["model"]["classes"], strides[i], config=config, device_id = config["device_id"]))

		self.layers = nn.ModuleList([LVnet_losses[i] for i in range(len(strides))])

	def forward(self, input, target=None):
		result = []
		for i in range(len(input)):
			loss_or_output_data = self.layers[i](input[i], target)
			result.append(loss_or_output_data)
		return result
