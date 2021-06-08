from models.loss.centernet_loss import centernet_Loss
import torch.nn as nn
import torch
class centernet_loss_module(nn.Module):
	def __init__(self, config, stride=4, nstack=2):
		super().__init__()
		self.nstack = nstack
		if nstack == 1:
			self.center_loss = centernet_Loss(config["model"]["classes"], stride, config=config, device_id= config["device_id"])
		elif nstack == 2:
			self.center_loss1 = centernet_Loss(config["model"]["classes"], stride, config=config, device_id= config["device_id"])
			self.center_loss2 = centernet_Loss(config["model"]["classes"], stride, config=config, device_id= config["device_id"])


	def forward(self, input, target=None):
		result = []
		if self.nstack == 1:
			cls_pred, txty_pred, twth_pred = input[0]
			center_loss_input = torch.cat((txty_pred, twth_pred, cls_pred), dim=1)
			result.append(self.center_loss(center_loss_input, target))

		elif self.nstack == 2:
			if target == None:
				cls_pred, txty_pred, twth_pred = input[0]
				center_loss_input = torch.cat((txty_pred, twth_pred, cls_pred), dim=1)
				result.append(self.center_loss2(center_loss_input, target))
			else:
				#input1
				cls_pred1, txty_pred1, twth_pred1 = input[0]
				center_loss_input1 = torch.cat((txty_pred1, twth_pred1, cls_pred1), dim=1)
				result1 = self.center_loss1(center_loss_input1, target)

				#intput2
				cls_pred2, txty_pred2, twth_pred2 = input[1]
				center_loss_input2 = torch.cat((txty_pred2, twth_pred2, cls_pred2), dim=1)
				result2 = self.center_loss2(center_loss_input2, target)

				result3 = []
				for sub_list in list(zip(result1, result2)):
					result3.append(sub_list[0] + sub_list[1])

				result.append(result3) #TODO:合并结果
		return result
