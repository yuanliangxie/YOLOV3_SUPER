import torch.nn as nn
import torch
from utils.utils_select_device import select_device
class LFFD_loss_module(nn.Module):
	def __init__(self, config):
		super(LFFD_loss_module, self).__init__()

		self.num_classes = config['model']['classes']
		self.bbox_attrs = 4 + self.num_classes#85
		self.strides = [4, 4, 8, 8, 16, 32, 32, 32]
		self.choose_feature_map = [i for i in range(len(self.strides))]
		self.continuous_face_scale=[[10,15], [15, 20], [20, 40], [40, 70], [70, 110], [110, 250], [250, 400], [400, 560]]
		self.ignore_threshold = 0.5
		self.lambda_xy = 1
		self.lambda_wh = 1
		self.lambda_cls = 1
		self.lambda_conf_tcls = 1
		self.bce_loss = nn.BCELoss(reduction='none')#交叉熵
		self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
		self.device = select_device(config["device_id"])
		self.config = config


		self.focal_loss_gama = 2
		self.focal_loss_alpha = 0.25

	def forward(self, input, target_tensor=None, n_obj = 0):
		result = []
		if target_tensor != None:
			assert n_obj != 0, "n_obj等于0，需要debug n_obj"

		#self.choose_feature_map = [0, 1] #test1
		#self.choose_feature_map = [2, 3, 4, 5] #test2
		#self.choose_feature_map = [6, 7] #test3

		#for i, scale_input in enumerate(input):

		for i in self.choose_feature_map:
			scale_input = input[i]

			bs = scale_input.size(0)
			in_h = scale_input.size(2)
			in_w = scale_input.size(3)
			stride_h = self.strides[i]
			stride_w = self.strides[i]
			prediction = scale_input.view(bs, self.bbox_attrs, in_h, in_w).permute(0, 2, 3, 1).contiguous()

			# Get outputs
			x = torch.sigmoid(prediction[..., 0])  # Center x
			y = torch.sigmoid(prediction[..., 1])  # Center y
			w = prediction[..., 2]  # Width
			h = prediction[..., 3]  # Height
			conf = torch.sigmoid(prediction[..., 4])  # Cls pred.

			# Calculate offsets for each grid
			grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
				bs, 1, 1).view(x.shape).to(self.device)
			grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
				bs, 1, 1).view(y.shape).to(self.device)

			pred_boxes = torch.stack([x + grid_x, y + grid_y,
									  torch.exp(w) * sum(self.continuous_face_scale[i])/(2*self.strides[i]),
									  torch.exp(h) * sum(self.continuous_face_scale[i])/(2*self.strides[i])], dim=-1)

			if target_tensor is not None:

				t_x = target_tensor[i][..., 0]
				t_y = target_tensor[i][..., 1]
				t_w = target_tensor[i][..., 2]
				t_h = target_tensor[i][..., 3]
				t_cls = target_tensor[i][..., 4]
				t_scale_area_weight = target_tensor[i][..., 5]

				t_x, t_y = t_x.to(self.device), t_y.to(self.device)
				t_w, t_h = t_w.to(self.device), t_h.to(self.device)
				t_cls, t_scale_area_weight = t_cls.to(self.device), t_scale_area_weight.to(self.device)

				loss_x = (t_scale_area_weight * self.bce_loss(x * t_cls, t_x)).sum()/n_obj
				loss_y = (t_scale_area_weight * self.bce_loss(y * t_cls, t_y)).sum()/n_obj
				loss_w = (t_scale_area_weight * self.smooth_l1(w * t_cls, t_w)).sum()/n_obj
				loss_h = (t_scale_area_weight * self.smooth_l1(h * t_cls, t_h)).sum()/n_obj
				loss_cls_conf = (self.focal_loss_alpha * (1-conf)**self.focal_loss_gama * t_cls * self.bce_loss(conf * t_cls, t_cls)).sum()/n_obj + \
		           ((1-self.focal_loss_alpha) * conf**self.focal_loss_gama * (1-t_cls) * self.bce_loss(conf * (1-t_cls), (1-t_cls) * 0.0)).sum()/n_obj #这里采用focal_loss进行编写
				loss = loss_x + loss_y + loss_w + loss_h + loss_cls_conf
				result.append([loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_cls_conf.item()])
			else:
				_scale = torch.FloatTensor([stride_w, stride_h] * 2).to(self.device)
				output = torch.cat((pred_boxes.detach().view(bs, -1, 4) * _scale,
									conf.view(bs, -1, 1), torch.ones_like(conf.view(bs, -1, 1))), -1) #仿造框架编写，其实只需要conf就够了
				result.append(output.data)
		return result