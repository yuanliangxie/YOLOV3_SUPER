import torch.nn as nn
import torch
from utils.utils_select_device import select_device
class LVnetloss_module(nn.Module):
	def __init__(self, config):
		super(LVnetloss_module, self).__init__()

		self.num_classes = config['model']['classes']
		self.bbox_attrs = 4 + self.num_classes#85
		self.strides = [16, 32]
		self.choose_feature_map = [i for i in range(len(self.strides))]
		self.lambda_xy = 1
		self.lambda_wh = 1
		self.lambda_cls = 1
		self.lambda_conf_tcls = 1
		self.bce_loss = nn.BCELoss(reduction='none')#交叉熵
		self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
		self.device = select_device(config["device_id"])
		self.config = config
		self.anchors = [[100, 65], [164, 121]] #(w, h)


		self.focal_loss_gama = 2
		self.focal_loss_alpha = 0.25

	def forward(self, pred_boxes_list, pred_cls_conf_list, pred_position_list, target_tensor=None, n_obj = 0):
		result = []
		if target_tensor != None:
			assert n_obj != 0, "n_obj等于0，需要debug n_obj"

		#self.choose_feature_map = [0] #test1
		#self.choose_feature_map = [3] #test2

		for i in self.choose_feature_map:

			pred_boxes = pred_boxes_list[i]
			conf = pred_cls_conf_list[i].squeeze(-1) #把后面的维去掉
			x, y, w, h = pred_position_list[i]

			bs = pred_boxes.size(0)
			stride_h = self.strides[i]
			stride_w = self.strides[i]

			if target_tensor is not None:

				t_x = target_tensor[i][..., 0]
				t_y = target_tensor[i][..., 1]
				t_w = target_tensor[i][..., 2]
				t_h = target_tensor[i][..., 3]
				t_cls = target_tensor[i][..., 4]
				t_scale_area_weight = target_tensor[i][..., 5]
				t_noobj_mask = target_tensor[i][..., 6]

				t_x, t_y = t_x.to(self.device), t_y.to(self.device)
				t_w, t_h = t_w.to(self.device), t_h.to(self.device)
				t_cls, t_scale_area_weight = t_cls.to(self.device), t_scale_area_weight.to(self.device)
				t_noobj_mask = t_noobj_mask.to(self.device)

				loss_x = (t_scale_area_weight * self.smooth_l1(x * t_cls, t_x)).sum()/n_obj
				loss_y = (t_scale_area_weight * self.smooth_l1(y * t_cls, t_y)).sum()/n_obj
				loss_w = (t_scale_area_weight * self.smooth_l1(w * t_cls, t_w)).sum()/n_obj
				loss_h = (t_scale_area_weight * self.smooth_l1(h * t_cls, t_h)).sum()/n_obj
				# loss_cls_conf = (self.focal_loss_alpha * (1-conf)**self.focal_loss_gama * t_cls * self.bce_loss(conf * t_cls, t_cls)).sum()/n_obj + \
				# 				((1-self.focal_loss_alpha) * conf**self.focal_loss_gama * (1-t_cls) * self.bce_loss(conf * (1-t_cls), (1-t_cls) * 0.0)).sum()/n_obj #这里采用focal_loss进行编写
				loss_cls_conf = self.bce_loss(t_cls * conf, t_cls).sum()/n_obj + 0.5*self.bce_loss(t_noobj_mask*conf, t_noobj_mask*0).sum()/n_obj
				loss = loss_x + loss_y + loss_w + loss_h + loss_cls_conf
				result.append([loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_cls_conf.item()])
			else:
				_scale = torch.FloatTensor([stride_w, stride_h] * 2).to(self.device)
				output = torch.cat((pred_boxes.detach().view(bs, -1, 4) * _scale,
									conf.view(bs, -1, 1), torch.ones_like(conf.view(bs, -1, 1))), -1) #仿造框架编写，其实只需要conf就够了
				result.append(output.data)
		return result