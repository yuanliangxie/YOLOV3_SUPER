import torch.nn as nn
import torch
import numpy as np
import cv2 as cv
def calculate_EPR(model): #TODO:尝试通过加载预训练权重计算有效感受野
	for module in model.modules():
		try:
			nn.init.constant_(module.weight, 0.05)
			nn.init.zeros_(module.bias)
			nn.init.zeros_(module.running_mean)
			nn.init.ones_(module.running_var)
		except Exception as e:
			pass
		if type(module) is nn.BatchNorm2d:
			module.eval()


	input = torch.ones(1, 3, 640, 640, requires_grad= True)
	model.zero_grad()
	features = model(input)

	for i in range(len(features)):
		# if i != len(features)-1:
		# 	continue
		x = features[i]
		#g_x = torch.zeros(size=[1, 1, x.shape[2], x.shape[3]])
		g_x = torch.zeros_like(x)
		h, w = g_x.shape[2]//2, g_x.shape[3]//2
		g_x[:, :, h, w] = 1
		x.backward(g_x, retain_graph = True)
		# x = torch.mean(x, 1, keepdim=True)
		# fake_fp = x * g_x[0, 0, ...]
		# fake_loss = torch.mean(fake_fp)
		# fake_loss.backward(retain_graph=True)

		show(input, i)
		model.zero_grad()
		input.grad.data.zero_()
		cv.waitKey(2000)
	cv.waitKey(0)

def cal_rf_wh(grad_input):
	binary_map: np.ndarray = (grad_input[:, :] > 0.0)

	x_cs: np.ndarray = binary_map.sum(-1) >= 1
	y_cs: np.ndarray = binary_map.sum(0) >= 1

	width = x_cs.sum()
	height = y_cs.sum()
	return (width, height)

def show(input, i):
	grad_input = np.abs(input.grad.data.numpy())
	grad_input = grad_input / np.max(grad_input)
	grad_input = grad_input.mean(0).mean(0)
	# 有效感受野 0.75 - 0.85
	#grad_input = np.where(grad_input > 0.85,1,0)
	#grad_input_ = np.where(grad_input > 0.75, 1, grad_input)


	# effient_values = grad_input > 0.0
	# samll_effient_values = grad_input <= 0.2
	# grad_input[np.logical_and(effient_values, samll_effient_values)] = 0.1
	#grad_input = grad_input * 100


	width, height = cal_rf_wh(grad_input)
	print("width:", width, "height:", height)

	grad_input_ERF = np.where(grad_input>0.01, 1, 0)
	width, height = cal_rf_wh(grad_input_ERF)
	print("ERF_width:", width, "ERF_height:", height)



	np.expand_dims(grad_input, axis=2).repeat(3, axis=2)
	grad_input = (grad_input * 255).astype(np.uint8)
	cv.imshow("receip_field"+str(i), grad_input)
	#cv.imwrite("./receip_field"+str(i)+".png", grad_input)

