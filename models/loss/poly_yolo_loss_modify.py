import torch
import torch.nn as nn
import numpy as np
import math
from utils.utils_old import bbox_ious as bbox_iou
from utils.utils_select_device import select_device
from models.bricks.tricks import Gaussian_weight as Gaussian_calculate
from utils.utils_old import focal_loss_gaussian_weight as focal_loss, HeatmapLoss
import torch.nn.functional as F




class YOLOLoss(nn.Module):
	def __init__(self, num_classes, stride, config, device_id, topk = 100):#([],80,(w,h))
		super(YOLOLoss, self).__init__()
		self.sigama = 8
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

	def forward(self, input, targets=None):
		bs = input.size(0)
		in_h = input.size(2)
		in_w = input.size(3)
		stride_h = self.stride
		stride_w = self.stride
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


		pred_boxes = torch.stack([x + grid_x, y + grid_y, torch.exp(w) * self.sigama, torch.exp(h) * self.sigama], dim=-1)

		if targets is not None:
			n_obj, mask, tx, ty, tw, th, tcls, coord_scale, Gaussian_weight = self.get_target(targets, in_w, in_h)
			mask,coord_scale = mask.to(self.device), coord_scale.to(self.device)
			tx, ty, tw, th = tx.to(self.device), ty.to(self.device), tw.to(self.device), th.to(self.device)
			tcls = tcls.to(self.device)
			Gaussian_weight = Gaussian_weight.to(self.device)

			loss_x = (coord_scale * self.bce_loss(x * mask, tx)).sum()/n_obj
			loss_y = (coord_scale * self.bce_loss(y * mask, ty)).sum()/n_obj
			loss_w = (coord_scale* self.smooth_l1(w * mask , tw )).sum()/n_obj
			loss_h = (coord_scale* self.smooth_l1(h * mask , th )).sum()/n_obj

			if tcls[tcls == 1].shape[0] == 0:
				loss_conf_tcls = torch.tensor(0).to(self.device)
			else:
				#loss_conf_tcls = focal_loss(Gaussian_weight)(pred_cls, tcls).sum()/n_obj
				loss_conf_tcls = HeatmapLoss()(pred_cls, Gaussian_weight).sum()/n_obj

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

			# #encode_topk_to_our_format_first_method
			# mask = torch.zeros(bs, in_h, in_w)
			# mask_pred_cls = torch.zeros(bs, in_h, in_w, self.num_classes)
			# for b in bs:
			# 	for i in range(self.top_k):
			# 		ind = topk_inds[b, i]
			# 		cls = topk_clses[b, i]
			# 		if ind // in_w and ind % in_w == 0:
			# 			row = ind // in_w - 1
			# 			cloumn = in_w - 1
			# 		else:
			# 			row = ind // in_w
			# 			cloumn = ind - (row* in_w) -1
			# 		mask[b, row, cloumn] = 1
			# 		mask_pred_cls[b, row, cloumn, cls] = 1
			#
			# trans_pred_boxes = pred_boxes[mask==1]
			# trans_conf = torch.ones(bs, self.top_k, 1)
			# trans_pred_cls = pred_cls[mask_pred_cls==1]

			# #encode_topk_to_our_format_second_method
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

	def get_target(self, target, in_w, in_h):
		n_obj = 0
		bs = target.shape[0]
		mask = torch.zeros(bs, in_h, in_w, requires_grad=False)  # (bs,13,13)
		scales = torch.zeros(bs, in_h, in_w, requires_grad=False)
		tx = torch.zeros(bs, in_h, in_w, requires_grad=False)
		ty = torch.zeros(bs, in_h, in_w, requires_grad=False)
		tw = torch.zeros(bs, in_h, in_w, requires_grad=False)
		th = torch.zeros(bs, in_h, in_w, requires_grad=False)
		tcls = torch.zeros(bs, in_h, in_w, self.num_classes, requires_grad=False)  # self.num_classes
		Gaussian_weight = torch.zeros(bs, in_h, in_w, self.num_classes, requires_grad=False)


		for b in range(bs):
			for t in range(target.shape[1]):
				if target[b, t].sum() == 0:
					continue
				n_obj += 1
				# Convert to position relative to box
				gx = target[b, t, 1] * in_w
				gy = target[b, t, 2] * in_h
				gw = target[b, t, 3] * in_w
				gh = target[b, t, 4] * in_h
				# Get grid box indices
				gi = int(gx)
				gj = int(gy)


				#如果此格子已经生成gt了，那么就将这个真实标注框跳过了，因为４倍下采样已经很小了,对车来说应该够了，这个可以统计一下。
				if tcls[b, gj, gi, int(target[b, t, 0])] == 1:
					n_obj -= 1
					continue
				# Masks
				mask[b, gj, gi] = 1
				scales[b, gj, gi] = 2 - target[b, t, 3] * target[b, t, 4]

				# Coordinates
				tx[b, gj, gi] = gx - gi
				ty[b, gj, gi] = gy - gj
				# Width and height
				tw[b, gj, gi] = math.log(gw / self.sigama + 1e-16)  # 返回去的tw还是要除以选定的anchor的w*in_w
				# 为什么这里要用math而不是用torch,因为这里只是求真实的值，是用于对比的值，而不是反传回去的值
				th[b, gj, gi] = math.log(gh / self.sigama + 1e-16)
				# object
				tcls[b, gj, gi, int(target[b, t, 0])] = 1  # 这里的target就表明label的类别标签是要从0开始的#int(target([b,t,0])

				#获取高斯权重
				box_gaussian_map, border = self.gaussian_cal.get_box_gaussian_map(gi, gj, gw, gh, in_w, in_h)
				top, left, bottom, right = border
				temp_weight = Gaussian_weight[b, top:bottom + 1, left:right + 1, int(target[b, t, 0])]
				Gaussian_weight[b, top:bottom + 1, left:right + 1, int(target[b, t, 0])] = torch.max(temp_weight, box_gaussian_map)

		return n_obj, mask, tx, ty, tw, th, tcls, scales, Gaussian_weight

if __name__ == '__main__':
	device = select_device(0)
	import train.params_init_voc as params_init
	config = params_init.TRAINING_PARAMS
	yololoss = YOLOLoss(config["yolo"]["anchors"][0],
						config["yolo"]["classes"], (416, 416), config_anchor=config["yolo"]["anchors"], device_id=0)
	f = open('../train/output.pkl', 'rb')
	import pickle
	data = pickle.load(f)
	f.close()
	input = torch.Tensor(data).to(device)
	target_input = [[[0, 0.498, 0.327, 0.997, 0.537]]]
	target_input = np.array(target_input)
	print(yololoss(input, target_input)[0])
	#result: loss_x = 0.0025
	#        loss_y = 0.0036
	#        loss_h = 0.0002
	#        loss_w = 0.0002
	#        loss_conf = 0.3714
	#        loss_cls = 0.6655
