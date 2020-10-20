import torch
import torch.nn as nn
import numpy as np
import math
from utils.utils_old import bbox_ious as bbox_iou
from utils.utils_select_device import select_device


class YOLOLoss(nn.Module):
	def __init__(self, num_classes, stride, config, device_id):#([],80,(w,h))
		super(YOLOLoss, self).__init__()
		self.sigama = 8
		self.num_classes = num_classes#80
		self.bbox_attrs = 5 + num_classes#85
		self.stride = stride
		self.ignore_threshold = 0.5
		self.lambda_xy = 1
		self.lambda_wh = 1
		self.lambda_cls = 1
		self.lambda_conf = 1
		self.lambda_tcls = 1
		self.bce_loss = nn.BCELoss(reduction='none')#交叉熵
		self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
		self.device = select_device(device_id)
		self.config = config

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
		conf = torch.sigmoid(prediction[..., 4])  # Conf
		pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

		# Calculate offsets for each grid
		grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
			bs, 1, 1).view(x.shape).to(self.device)
		grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
			bs, 1, 1).view(y.shape).to(self.device)


		pred_boxes = torch.stack([x + grid_x, y + grid_y, torch.exp(w) * self.sigama, torch.exp(h) * self.sigama], dim=-1)

		if targets is not None:
			n_obj, mask, noobj_mask, tx, ty, tw, th, tcls, coord_scale = self.get_target(targets, in_w, in_h, pred_boxes.detach())
			mask, coord_scale = mask.to(self.device), coord_scale.to(self.device)
			tx, ty, tw, th = tx.to(self.device), ty.to(self.device), tw.to(self.device), th.to(self.device)
			tcls = tcls.to(self.device)
			noobj_mask = noobj_mask.to(self.device)

			loss_x = (coord_scale[mask==1] * self.bce_loss(x[mask==1], tx[mask==1])).sum()/n_obj
			loss_y = (coord_scale[mask==1] * self.bce_loss(y[mask==1], ty[mask==1])).sum()/n_obj
			loss_w = (coord_scale[mask==1] * self.smooth_l1(w[mask==1], tw[mask==1] )).sum()/n_obj
			loss_h = (coord_scale[mask==1] * self.smooth_l1(h[mask==1], th[mask==1] )).sum()/n_obj

			# loss_x = (coord_scale * self.bce_loss(x * mask, tx)).sum()/n_obj
			# loss_y = (coord_scale * self.bce_loss(y * mask, ty)).sum()/n_obj
			# loss_w = (coord_scale * self.smooth_l1(w * mask , tw )).sum()/n_obj
			# loss_h = (coord_scale * self.smooth_l1(h * mask , th )).sum()/n_obj


			loss_conf = self.bce_loss(conf * mask, mask).sum()/n_obj + 0.5 * self.bce_loss(noobj_mask * conf, noobj_mask*0).sum()/n_obj

			if tcls[tcls == 1].shape[0] == 0:#防止图片中没有物体进行预测时，这时候tcls则是为0
				loss_tcls = torch.tensor(0).to(self.device)
			else:
				loss_tcls = self.bce_loss(pred_cls[mask==1], tcls[mask==1]).sum()/n_obj


			loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
				   loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
				   loss_conf * self.lambda_conf + loss_tcls * self.lambda_tcls

			# print(
			#     'loss:{:.2f},loss_x:{:.5f},loss_y:{:.5f},loss_w:{:.5f},loss_h:{:.5f},loss_conf:{:.5f}'.format(
			#         loss, loss_x * self.lambda_xy, loss_y * self.lambda_xy, loss_w * self.lambda_wh,
			#               loss_h * self.lambda_wh, loss_conf_tcls * self.lambda_conf_tcls
			#     ))
			return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
				   loss_h.item(), loss_conf.item(), loss_tcls.item() #这里返回的只有loss没有item,因为loss还要反向传播
		else:
			# Results
			_scale = torch.FloatTensor([stride_w, stride_h] * 2).to(self.device)
			output = torch.cat((pred_boxes.reshape(bs, -1, 4) * _scale, conf.reshape(bs, -1, 1), pred_cls.reshape(bs, -1, self.num_classes)), -1)
			return output.data

	def get_target(self, target, in_w, in_h, pred_boxes):
		n_obj = 0
		bs = target.shape[0]
		mask = torch.zeros(bs, in_h, in_w, requires_grad=False)  # (bs,13,13)
		scales = torch.zeros(bs, in_h, in_w, requires_grad=False)
		tx = torch.zeros(bs, in_h, in_w, requires_grad=False)
		ty = torch.zeros(bs, in_h, in_w, requires_grad=False)
		tw = torch.zeros(bs, in_h, in_w, requires_grad=False)
		th = torch.zeros(bs, in_h, in_w, requires_grad=False)
		tcls = torch.zeros(bs, in_h, in_w, self.num_classes, requires_grad=False)  # self.num_classes
		noobj_mask = torch.ones(bs, in_h, in_w, requires_grad=False)



		for b in range(bs):
			pred_box = pred_boxes[b].view(-1, 4)
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
				gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0).to(self.device)

				#如果此格子已经生成gt了，那么就将这个真实标注框跳过了，因为４倍下采样已经很小了,对车来说应该够了，这个可以统计一下。
				if mask[b, gj, gi] == 1:
					n_obj -= 1
					continue
				# Masks
				pred_ious = bbox_iou(gt_box, pred_box).view(in_h, in_w)
				noobj_mask[b, pred_ious >= self.ignore_threshold] = 0
				mask[b, gj, gi] = 1
				noobj_mask[b, gj, gi] = 0
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

		return n_obj, mask, noobj_mask, tx, ty, tw, th, tcls, scales

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
