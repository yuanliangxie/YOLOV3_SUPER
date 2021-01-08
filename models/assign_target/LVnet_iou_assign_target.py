import math
import torch
import torch.nn as nn
from models.bricks.tricks import Gaussian_weight as Gaussian_calculate
from tools.time_analyze import func_line_time
import numpy as np
from utils.utils_old import bbox_ious as bbox_iou
from utils.utils_old import bbox_iou_xywh_numpy
from utils.utils_select_device import select_device

class assign_targets(object):
	def __init__(self, input_shape, device_id):
		"""
		:param target: b,t,5  [cls, x, y, w, h]
		:param input_shape: (h,w)
		"""
		self.oringin_h = input_shape[0]
		self.oringin_w = input_shape[1]
		self.strides = [4, 8, 16, 32]
		self.in_h = [input_shape[0]/s for s in self.strides]
		self.in_w = [input_shape[1]/s for s in self.strides]
		self.continuous_face_scale=[[15, 45], [45, 75], [75, 135], [135, 260]]
		self.anchors = [[28, 21], [57, 38], [100, 65], [164, 121]] #(w, h)
		self.num_classes = 1
		self.bbox_attrs = 4 + self.num_classes
		self.gaussian_cal = Gaussian_calculate(min_overlap=0.7)
		self.device = select_device(device_id)
		self.ignore_threshold = 0.7



	#@func_line_time
	def __call__(self, target, loss_inputs_detach):
		pred_boxes_list, pred_cls_conf_list = self.get_pred_boxes(loss_inputs_detach)
		if target == None:
			return None, 0
		self.target = target
		bs = self.target.shape[0]
		ts = self.target.shape[1]
		self.t_scales = []
		for i, strides in enumerate(self.strides[0:2]):
			self.t_scale = torch.zeros((bs, int(self.in_h[i]), int(self.in_w[i]), 7), requires_grad=False)
			self.noobj_mask = torch.ones((bs, int(self.in_h[i]), int(self.in_w[i]), 1), requires_grad=False)
			self.t_scale = torch.cat([self.t_scale, self.noobj_mask], dim=-1)
			self.t_scales.append(self.t_scale)
		for i, strides in enumerate(self.strides[2:]):
			i = i+2
			self.t_scale = torch.zeros((bs, int(self.in_h[i]), int(self.in_w[i]), 6), requires_grad=False)
			self.noobj_mask = torch.ones((bs, int(self.in_h[i]), int(self.in_w[i]), 1), requires_grad=False)
			self.t_scale = torch.cat([self.t_scale, self.noobj_mask], dim=-1)
			self.t_scales.append(self.t_scale)

		n_obj = 0
		for b in range(bs):
			pred_boxes_b = [pred_boxes[b, :].reshape(-1, 4) for pred_boxes in pred_boxes_list]
			for t in range(ts):
				if self.target[b, t].sum() == 0:
					continue
				else:

					n_obj = n_obj + 1
					x = [self.target[b, t, 1] * in_w for in_w in self.in_w]
					y = [self.target[b, t, 2] * in_h for in_h in self.in_h]
					w = [self.target[b, t, 3] * in_w for in_w in self.in_w]
					h = [self.target[b, t, 4] * in_h for in_h in self.in_h]
					scale_xywh = list(zip(x, y, w, h))
					gt_boxes = [0,
								0,
								self.target[b, t, 3]*self.oringin_w,
								self.target[b, t, 4]*self.oringin_h]
					self.matching_strategy(b, t, scale_xywh, gt_boxes, pred_boxes_b)

		return self.t_scales, n_obj#, pred_boxes_list, pred_cls_conf_list



	def get_pred_boxes(self, loss_input_detach):
		pred_boxes_list = []
		pred_cls_conf_list = []
		for i, input in enumerate(loss_input_detach):
			bs = input.size(0)
			in_h = input.size(2)
			in_w = input.size(3)
			stride_h = self.strides[i]
			stride_w = self.strides[i]
			prediction = input.view(bs,	self.bbox_attrs, in_h, in_w).permute(0, 2, 3, 1).contiguous()

			# Get outputs
			x = torch.sigmoid(prediction[..., 0])  # Center x
			y = torch.sigmoid(prediction[..., 1])  # Center y
			w = prediction[..., 2]  # Width
			h = prediction[..., 3]  # Height
			#conf = torch.sigmoid(prediction[..., 4])  # Conf
			pred_cls = torch.sigmoid(prediction[..., 4:])  # Cls pred.

			# Calculate offsets for each grid
			grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
				bs, 1, 1).view(x.shape).to(self.device)
			grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
				bs, 1, 1).view(y.shape).to(self.device)
			pred_boxes = torch.stack([x + grid_x, y + grid_y, torch.exp(w) * self.anchors[i][0]/stride_w, torch.exp(h) * self.anchors[i][1]/stride_h], dim=-1)
			pred_boxes_list.append(pred_boxes)
			pred_cls_conf_list.append(pred_cls)
		return pred_boxes_list, pred_cls_conf_list

	#@func_line_time
	def matching_strategy(self, b, t, scale_xywh, gt_boxes, pred_boxes_b):
		gt_boxes = np.array(gt_boxes)
		scale_xywh = np.array(scale_xywh)
		anchor_shapes = np.concatenate((np.array([0, 0]*4).reshape(4, 2), np.array(self.anchors)), 1)
		anchor_match_ious = bbox_iou_xywh_numpy(gt_boxes.reshape(1, -1), anchor_shapes)
		best_n = np.argmax(anchor_match_ious)
		if best_n < 2:
			self.assign_centernet_scale(scale_xywh, pred_boxes_b, b, t, best_index=best_n)
		else:
			self.assign_scale(scale_xywh, pred_boxes_b, b, t, best_index=best_n)

	def assign_scale(self, scale_xywh, pred_boxes_b, b, t, best_index):

		self.assign_noobj_mask_all_scale_feature_map(scale_xywh, pred_boxes_b, b)

		gx, gy, gw, gh = scale_xywh[best_index]

		# Get grid box indices
		gi = int(gx)
		gj = int(gy)

		self.t_scales[best_index][b, gj, gi, 0] = gx - gi
		self.t_scales[best_index][b, gj, gi, 1] = gy - gj
		self.t_scales[best_index][b, gj, gi, 2] = math.log(gw / (self.anchors[best_index][0]/(self.strides[best_index])) + 1e-16)
		self.t_scales[best_index][b, gj, gi, 3] = math.log(gh / (self.anchors[best_index][1]/(self.strides[best_index])) + 1e-16)

		self.t_scales[best_index][b, gj, gi, 4] = 1 #cls

		self.t_scales[best_index][b, gj, gi, 5] = 2 - self.target[b, t, 3] * self.target[b, t, 4] #scale_mask

		self.t_scales[best_index][b, gj, gi, 6] = 0

		return 0

	#去除层之间的干扰
	def assign_noobj_mask_all_scale_feature_map(self, scale_xywh, pred_boxes_b, b):
		for index, xywh in enumerate(scale_xywh):
			gx, gy, gw, gh = xywh
			gt_box = torch.FloatTensor([[gx, gy, gw, gh]]).to(self.device)
			pred_ious = bbox_iou(gt_box, pred_boxes_b[index]).reshape(int(self.in_h[index]), int(self.in_w[index]))
			self.t_scales[index][..., -1][b, pred_ious >= self.ignore_threshold] = 0


	def assign_centernet_scale(self, scale_xywh, pred_boxes_b, b, t, best_index):

		self.assign_noobj_mask_all_scale_feature_map(scale_xywh, pred_boxes_b, b)

		bs = self.target.shape[0]
		Gaussian_weight = torch.zeros(bs, int(self.in_h[best_index]), int(self.in_w[best_index]), self.num_classes, requires_grad=False)
		#t_cls = torch.zeros(bs, self.in_h[scale_index], self.in_w[scale_index], self.num_classes, requires_grad=False)

		gx, gy, gw, gh = scale_xywh[best_index]

		# Get grid box indices
		gi = int(gx)
		gj = int(gy)

		t_scale = self.t_scales[best_index]
		t_scale[b, gj, gi, 0] = gx - gi
		t_scale[b, gj, gi, 1] = gy - gj
		t_scale[b, gj, gi, 2] = math.log(gw / (self.anchors[best_index][0]/(self.strides[best_index])) + 1e-16)
		t_scale[b, gj, gi, 3] = math.log(gh / (self.anchors[best_index][1]/(self.strides[best_index])) + 1e-16)

		t_scale[b, gj, gi, 4] = 1 #cls

		t_scale[b, gj, gi, 5] = 2 - self.target[b, t, 3] * self.target[b, t, 4] #scale_mask

		#获取高斯权重
		box_gaussian_map, border = self.gaussian_cal.get_box_gaussian_map(gi, gj, gw, gh, int(self.in_w[best_index]), int(self.in_h[best_index]))
		top, left, bottom, right = border
		temp_weight = Gaussian_weight[b, top:bottom + 1, left:right + 1, int(self.target[b, t, 0])]
		Gaussian_weight[b, top:bottom + 1, left:right + 1, int(self.target[b, t, 0])] = torch.max(temp_weight, box_gaussian_map)
		t_scale[..., 6:7] = Gaussian_weight

		#noobj_mask
		t_scale[b, gj, gi, 7] = 0