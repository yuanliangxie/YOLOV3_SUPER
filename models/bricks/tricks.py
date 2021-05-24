import torch.nn as nn
import torch
import numpy as np
__all__ = ['label_smooth', 'mix_up', 'GIOU']


class label_smooth(object):
	def __init__(self, theta=0.1, classes=20):
		self.theta = theta
		self.classes = classes
		self.k = self.theta / (self.classes - 1)
	def smooth(self, tcls, mask):
		#k = self.theta / (self.classes - 1)
		tcls[mask == 1] *= (1 - self.theta - self.k)
		tcls[mask == 1] += self.k
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



class Gaussian_weight:
	def __init__(self, min_overlap):
		self.min_overlap = 0.7

	def gaussian_radius_official(self, gw, gh, min_overlap=0.7):
		"""
		:param det_size: 4倍下采样的[w, h]
		:param min_overlap:
		:return:
		"""
		box_h, box_h = gw, gh
		a1 = 1
		b1 = (box_h + box_h)
		c1 = box_h * box_h * (1 - min_overlap) / (1 + min_overlap)
		sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
		r1 = (b1 + sq1) / 2  # (2*a1)

		a2 = 4
		b2 = 2 * (box_h + box_h)
		c2 = (1 - min_overlap) * box_h * box_h
		sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
		r2 = (b2 + sq2) / 2  # (2*a2)

		a3 = 4 * min_overlap
		b3 = -2 * min_overlap * (box_h + box_h)
		c3 = (min_overlap - 1) * box_h * box_h
		sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
		r3 = (b3 + sq3) / 2  # (2*a3)

		r = min(r1, r2, r3)

		return r, r

	def get_gaussian_radius(self, gw, gy):
		"""
		:param det_size: 4倍下采样的真实标注框的[w, h], 可以为浮点数
		:return:
		"""
		box_w, box_h = gw, gy
		a1 = box_h / box_w
		b1 = -(box_h + box_h)
		c1 = box_w * box_h * (1 - self.min_overlap) / (1 + self.min_overlap)
		sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
		r_x = (-b1 - sq1) / (2 * a1)  # (2*a1)
		r_y = r_x * (box_h / box_w)
		return r_x, r_y



	def get_box_gaussian_map(self, gi, gj, gw, gh, in_w, in_h):
		"""
		:param gi: the position of the positive sample in x_axis. dtype:int
		:param gj: the position of the positive sample in y_axis. dtype:int
		:param gw: the width of the groundtruth box.              dtype:gw
		:param gh: the height of the groundtruth box.             dtype:gh
		:param in_w: the width of the feature map.                dtype:int
		:param in_h: the height of the feature map.               dtype:int
		:return: the gaussian weight map, shape: [in_w, in_h].
		"""
		#r_x, r_y = self.get_gaussian_radius(gw, gh)
		r_x, r_y = self.gaussian_radius_official(gw, gh)
		omiga_x = r_x / 3
		omiga_y = r_y / 3
		x_i = torch.arange(in_w).repeat(in_h, 1).float()
		y_j = torch.arange(in_h).repeat(in_w, 1).permute(1, 0).float()
		position_x = torch.ones(x_i.shape) * gi
		position_y = torch.ones(y_j.shape) * gj
		x_direct = torch.pow(x_i - position_x, 2) / (omiga_x**2)
		y_direct = torch.pow(y_j - position_y, 2) / (omiga_y**2)
		entire_gaussian_map_render = torch.exp(-0.5 * (x_direct + y_direct))

		# 对边界进行约束，防止越界
		left_r, right_r = min(gi+1, r_x), min(in_w - gi - 1, r_x)
		top_r, bottom_r = min(gj+1, r_y), min(in_h - gj - 1, r_y)


		left, right = int(gi - left_r), int(gi + right_r)
		top, bottom = int(gj - top_r), int(gj + bottom_r)


		temp = entire_gaussian_map_render[top:bottom+1, left:right+1]
		border = [top, left, bottom, right]
		return temp, border





if __name__ == '__main__':

	xie_gaussian = Gaussian_weight(0.7)
	box_gaussian_map, border = xie_gaussian.get_box_gaussian_map(300, 300, 150, 100, 608, 608)
	Gaussian_weight = torch.zeros(608, 608)
	xie_gaussian.update_Gaussian_weight(Gaussian_weight, box_gaussian_map, border)
	A = Gaussian_weight.numpy()
	import cv2
	import numpy as np
	#A = cv2.rectangle(A, (250, 225), (350, 375), color=(255, 255, 0), thickness=2)
	#A.astype(np.uint8)
	cv2.imshow('pic', A)
	cv2.waitKey(0)












# def gaussian_radius_xie(det_size, min_overlap=0.7):
# 	"""
# 	:param det_size: 4倍下采样的真实标注框的[w, h]
# 	:param min_overlap:
# 	:return:
# 	"""
# 	box_w, box_h  = det_size
# 	a1 = box_h/box_w
# 	b1 = -(box_h + box_h)
# 	c1 = box_w * box_h * (1 - min_overlap) / (1 + min_overlap)
# 	sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
# 	r_x = (-b1 - sq1) / (2*a1) #(2*a1)
# 	r_y = r_x * (box_h/box_w)
# 	return r_x, r_y






