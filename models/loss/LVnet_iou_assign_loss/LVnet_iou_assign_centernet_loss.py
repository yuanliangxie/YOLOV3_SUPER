import torch
import torch.nn as nn
import numpy as np
import math
from utils.utils_old import bbox_ious as bbox_iou
from utils.utils_select_device import select_device
from models.bricks.tricks import Gaussian_weight as Gaussian_calculate
import torch.nn.functional as F

class HeatmapLoss(nn.Module):
	def __init__(self,  weight=None, alpha=2, beta=4, reduction='mean'):
		super(HeatmapLoss, self).__init__()
		self.alpha = alpha
		self.beta = beta
	def forward(self, inputs, targets, noobj_mask):
		#inputs = torch.sigmoid(inputs)
		center_id = (targets == 1.0).float()
		other_id = (targets != 1.0).float()
		center_loss = -center_id * (1.0 - inputs) ** self.alpha * torch.log(inputs + 1e-14)
		other_loss = -other_id * noobj_mask * (1 - targets) ** self.beta * (inputs) ** self.alpha * torch.log(1.0 - inputs + 1e-14)
		return center_loss + other_loss




class centernet_Loss(nn.Module):
	def __init__(self, num_classes, stride, anchor, config, device_id, topk = 100):#([],80,(w,h))
		super(centernet_Loss, self).__init__()
		self.sigama = 8
		self.anchor = anchor
		self.num_classes = num_classes#80
		self.bbox_attrs = 4 + num_classes#85
		self.stride = stride
		self.ignore_threshold = 0.5
		self.lambda_xy = 1
		self.lambda_wh = 1
		self.lambda_cls = 1
		self.lambda_conf_tcls = 1
		self.bce_loss = nn.BCELoss(reduction='none')#交叉熵
		self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
		self.device = select_device(device_id)
		self.gaussian_cal = Gaussian_calculate(min_overlap=0.7)
		self.config = config
		self.topk = topk

	def forward(self, pred_boxes, pred_cls_conf, pred_position, targets=None, n_obj=None):
		bs = pred_boxes.size(0)
		stride_h = self.stride
		stride_w = self.stride
		pred_cls = pred_cls_conf
		x, y, w, h = pred_position
		if targets is not None:

			tx = targets[..., 0]
			ty = targets[..., 1]
			tw = targets[..., 2]
			th = targets[..., 3]
			mask = targets[..., 4]
			tcls = targets[..., 4:5] #shape(bs, in_h ,in_w, 1)
			coord_scale = targets[..., 5]
			Gaussian_weight = targets[..., 6:7] #shape(bs, in_h, in_w, 1)
			noobj_mask = targets[..., 7:8]

			mask, coord_scale = mask.to(self.device), coord_scale.to(self.device)
			tx, ty, tw, th = tx.to(self.device), ty.to(self.device), tw.to(self.device), th.to(self.device)
			tcls = tcls.to(self.device)
			Gaussian_weight = Gaussian_weight.to(self.device)
			noobj_mask = noobj_mask.to(self.device)

			loss_x = (coord_scale * self.smooth_l1(x * mask, tx)).sum()/n_obj
			loss_y = (coord_scale * self.smooth_l1(y * mask, ty)).sum()/n_obj
			loss_w = (coord_scale* self.smooth_l1(w * mask , tw )).sum()/n_obj
			loss_h = (coord_scale* self.smooth_l1(h * mask , th )).sum()/n_obj

			if tcls[tcls == 1].shape[0] == 0:
				loss_conf_tcls = torch.tensor(0).to(self.device)
			else:
				#loss_conf_tcls = focal_loss(Gaussian_weight)(pred_cls, tcls).sum()/n_obj
				loss_conf_tcls = HeatmapLoss()(pred_cls, Gaussian_weight, noobj_mask).sum()/n_obj

			loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
				   loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
				   loss_conf_tcls * self.lambda_conf_tcls

			# print(
			#     'loss:{:.2f},loss_x:{:.5f},loss_y:{:.5f},loss_w:{:.5f},loss_h:{:.5f},loss_conf:{:.5f}'.format(
			#         loss, loss_x * self.lambda_xy, loss_y * self.lambda_xy, loss_w * self.lambda_wh,
			#               loss_h * self.lambda_wh, loss_conf_tcls * self.lambda_conf_tcls
			#     ))
			return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
				   loss_h.item(), loss_conf_tcls.item() #这里返回的只有loss没有item,因为loss还要反向传播
		else:

			hmax = F.max_pool2d(pred_cls, kernel_size=5, padding=2, stride=1)
			keep = (hmax == pred_cls).float()
			pred_cls *= keep
			# topk
			topk_scores, topk_inds, topk_clses = self._topk(pred_cls)
			pred_boxes = pred_boxes.reshape(bs, -1, 4)
			pred_cls = pred_cls.reshape(bs, -1, self.num_classes)

			trans_pred_boxes = torch.zeros(bs, self.topk, 4).to(self.device)
			trans_pred_cls = torch.zeros(bs, self.topk, self.num_classes).to(self.device)
			for b in range(bs):
				trans_pred_boxes[b] = pred_boxes[b, topk_inds[b, :], :]
				trans_pred_cls[b] = pred_cls[b, topk_inds[b, :], :]
			trans_conf = torch.ones(bs, self.topk, 1).to(self.device)



			# Results
			_scale = torch.FloatTensor([stride_w, stride_h] * 2).to(self.device)
			output = torch.cat((trans_pred_boxes.detach() * _scale, trans_conf, trans_pred_cls.detach()), -1)
			return output.data

	def _gather_feat(self, feat, ind, mask=None):
		dim = feat.size(2)
		ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
		feat = feat.gather(1, ind)
		if mask is not None:
			mask = mask.unsqueeze(2).expand_as(feat)
			feat = feat[mask]
			feat = feat.view(-1, dim)
		return feat

	def _topk(self, scores):
		B, H, W, C = scores.size()

		scores = scores.permute(0, 3, 1, 2).contiguous()

		topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), self.topk)

		topk_inds = topk_inds % (H * W)

		topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), self.topk)

		topk_clses = (topk_ind / self.topk).int()

		topk_inds = self._gather_feat(topk_inds.view(B, -1, 1), topk_ind).view(B, self.topk)

		return topk_score, topk_inds, topk_clses


