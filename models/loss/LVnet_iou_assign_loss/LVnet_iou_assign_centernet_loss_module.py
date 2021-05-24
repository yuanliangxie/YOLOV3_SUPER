from models.loss.LVnet_iou_assign_loss.LVnet_iou_assign_centernet_loss import centernet_Loss
import torch.nn as nn
class centernet_loss_module(nn.Module):
	def __init__(self, config, anchor, stride=4):
		super().__init__()
		self.centernet_loss = centernet_Loss(config["model"]["classes"], stride, anchor, config=config, device_id=config["device_id"])
	def forward(self, pred_boxes, pred_cls_conf, pred_position, target=None, n_obj=None):
		result = []
		result.append(self.centernet_loss(pred_boxes, pred_cls_conf, pred_position, target, n_obj))
		return result
