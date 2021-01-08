import math
import torch
from models.bricks.tricks import Gaussian_weight as Gaussian_calculate
from tools.time_analyze import func_line_time

class assign_targets(object):
	def __init__(self, input_shape):
		"""
		:param target: b,t,5  [cls, x, y, w, h]
		:param input_shape: (h,w)
		"""
		self.strides = [4, 8, 16, 32]
		self.in_h = [input_shape[0]/s for s in self.strides]
		self.in_w = [input_shape[1]/s for s in self.strides]
		self.continuous_face_scale=[[15, 45], [45, 75], [75, 135], [135, 260]]
		self.anchors = [[28, 21], [57, 38], [100, 65], [164, 121]] #(w, h)
		self.num_classes = 1
		self.gaussian_cal = Gaussian_calculate(min_overlap=0.7)

	#@func_line_time
	def __call__(self, target):
		if target == None:
			return None, 0

		self.target = target
		bs = self.target.shape[0]
		ts = self.target.shape[1]
		self.t_scales = []
		for i, strides in enumerate(self.strides[0:2]):
			self.t_scale = torch.zeros((bs, int(self.in_h[i]), int(self.in_w[i]), 7), requires_grad=False)
			self.t_scales.append(self.t_scale)
		for i, strides in enumerate(self.strides[2:]):
			i = i+2
			self.t_scale = torch.zeros((bs, int(self.in_h[i]), int(self.in_w[i]), 6), requires_grad=False)
			self.t_scales.append(self.t_scale)

		n_obj = 0
		for b in range(bs):
			for t in range(ts):
				if self.target[b, t].sum() == 0:
					continue
				else:
					n_obj = n_obj + 1
					x = [self.target[b, t, 1] * in_w for in_w in self.in_w]
					y = [self.target[b, t, 2] * in_h for in_h in self.in_h]
					w = [self.target[b, t, 3] * in_w for in_w in self.in_w]
					h = [self.target[b, t, 4] * in_h for in_h in self.in_h]

					self.matching_strategy(b, t, x, y, w, h)

		return self.t_scales, n_obj
	#@func_line_time
	def matching_strategy(self, b, t, x, y, w, h):
		scale_xywh = list(zip(x, y, w, h))
		area = [couple[0] *couple[1] for couple in zip(w, h)]
		for i,  scale in enumerate(self.continuous_face_scale):
			if scale[0]*scale[0]/(self.strides[i]**2) <= area[i] <= scale[1]*scale[1]/(self.strides[i]**2):
				if i < 2:
					self.assign_centernet_scale(self.t_scales[i], scale_xywh[i], b, t, scale_index=i)
				else:
					self.assign_scale(self.t_scales[i], scale_xywh[i], b, t, scale_index=i) #为某层赋予真实标签！
				continue
			elif i== (len(self.continuous_face_scale)-1) and area[i] > scale[1]*scale[1]/(self.strides[i]**2):
				self.assign_scale(self.t_scales[i], scale_xywh[i], b, t, scale_index=i) #如果annotation_area get more than biggets scale


	def assign_scale(self, t_scale, scale_xywh, b, t, scale_index):

		gx, gy, gw, gh = scale_xywh

		# Get grid box indices
		gi = int(gx)
		gj = int(gy)

		t_scale[b, gj, gi, 0] = gx - gi
		t_scale[b, gj, gi, 1] = gy - gj
		t_scale[b, gj, gi, 2] = math.log(gw / (self.anchors[scale_index][0]/(self.strides[scale_index])) + 1e-16)
		t_scale[b, gj, gi, 3] = math.log(gh / (self.anchors[scale_index][1]/(self.strides[scale_index])) + 1e-16)

		t_scale[b, gj, gi, 4] = 1 #cls

		t_scale[b, gj, gi, 5] = 2 - self.target[b, t, 3] * self.target[b, t, 4] #scale_mask

		return 0

	def assign_centernet_scale(self, t_scale, scale_xywh, b, t, scale_index):
		bs = self.target.shape[0]
		Gaussian_weight = torch.zeros(bs, int(self.in_h[scale_index]), int(self.in_w[scale_index]), self.num_classes, requires_grad=False)
		#t_cls = torch.zeros(bs, self.in_h[scale_index], self.in_w[scale_index], self.num_classes, requires_grad=False)

		gx, gy, gw, gh = scale_xywh

		# Get grid box indices
		gi = int(gx)
		gj = int(gy)

		t_scale[b, gj, gi, 0] = gx - gi
		t_scale[b, gj, gi, 1] = gy - gj
		t_scale[b, gj, gi, 2] = math.log(gw / (self.anchors[scale_index][0]/(self.strides[scale_index])) + 1e-16)
		t_scale[b, gj, gi, 3] = math.log(gh / (self.anchors[scale_index][1]/(self.strides[scale_index])) + 1e-16)

		t_scale[b, gj, gi, 4] = 1 #cls

		t_scale[b, gj, gi, 5] = 2 - self.target[b, t, 3] * self.target[b, t, 4] #scale_mask

		#获取高斯权重
		box_gaussian_map, border = self.gaussian_cal.get_box_gaussian_map(gi, gj, gw, gh, int(self.in_w[scale_index]), int(self.in_h[scale_index]))
		top, left, bottom, right = border
		temp_weight = Gaussian_weight[b, top:bottom + 1, left:right + 1, int(self.target[b, t, 0])]
		Gaussian_weight[b, top:bottom + 1, left:right + 1, int(self.target[b, t, 0])] = torch.max(temp_weight, box_gaussian_map)
		t_scale[..., 6:] = Gaussian_weight