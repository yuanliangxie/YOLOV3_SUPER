import torch.nn as nn
import torch
import numpy as np
__all__ = ['label_smooth', 'mix_up', 'GIOU']


class label_smooth(object):
	def __init__(self, theta=0.1, classes=20):
		self.theta = theta
		self.classes = classes
	def smooth(self, tcls, mask):
		k = self.theta / (self.classes - 1)
		tcls[mask == 1] *= (1 - self.theta - k)
		tcls[mask == 1] += k
		return tcls

class mix_up(object):
	def __init__(self, alpha):
		self.alpha = alpha

	def mixup(self, images, targets):
		index = torch.randperm(images.size(0))
		lam = 0.5#np.random.beta(self.alpha, self.alpha)
		output_images = lam * images + (1- lam) * images[index, :]
		targets = torch.cat((targets, targets[index, :]), 1)
		return output_images, targets

class GIOU(object):
	def __init__(self):
		pass

	def cal_giou_loss(self, gt_boxes, pred_boxes, mask):
		"""

		:param gt_boxes: shape(b, anchor_index, h, w, 4)  [x_c, y_c, w, h]
		:param pred_boxes: shape(b, anchor_index, h, w, 4)  [x_c, y_c, w, h]
		:param mask: shape(b, anchor_index, h, w)
		:return: giou_loss
		"""
		gt_boxes = gt_boxes[mask == 1] #shape [N, 4]
		pred_boxes = pred_boxes[mask == 1] #shape[N, 4]

		gt_boxes[:, :2] = gt_boxes[:, :2] - gt_boxes[:, 2:]/2
		gt_boxes[:, 2:] = gt_boxes[:, :2] + gt_boxes[:, 2:]

		pred_boxes[:, :2] = pred_boxes[:, :2] - pred_boxes[:, 2:]/2
		pred_boxes[:, 2:] = pred_boxes[:, :2] + pred_boxes[:, 2:]

		left_up = torch.min(gt_boxes[:, :2], pred_boxes[:, :2])
		right_bottom = torch.max(gt_boxes[:, 2:], pred_boxes[:, 2:])

		inter_left_up = torch.max(gt_boxes[:, :2], pred_boxes[:, :2])
		inter_right_bottom = torch.min(gt_boxes[:, 2:], pred_boxes[:, 2:])

		inter_area = torch.prod(torch.clamp(inter_right_bottom-inter_left_up, min=0), dim=1)

		gt_boxes_area = torch.prod(torch.clamp(gt_boxes[:, 2:] - gt_boxes[:, :2], min=0), dim=1)
		pred_boxes_area = torch.prod(torch.clamp(pred_boxes[:, 2:] - pred_boxes[:, :2], min=0), dim=1)

		IOU = inter_area/(gt_boxes_area + pred_boxes_area - inter_area + 1e-7)

		C_boxes_area = torch.prod(torch.clamp(right_bottom-left_up, min=0), dim=1)

		giou= IOU - (C_boxes_area + inter_area - gt_boxes_area - pred_boxes_area)/(C_boxes_area + 1e-7)

		return 1-giou






