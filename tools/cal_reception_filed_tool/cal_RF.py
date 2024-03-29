import torch
import torchsummaryX as summaryX
from collections import OrderedDict
import torch.nn as nn
from .pytorch import PytorchReceptiveField


def get_names_dict(model):
	"""Recursive walk to get names including path."""
	names = {}

	def _get_names(module, parent_name=""):
		for key, m in module.named_children():
			cls_name = str(m.__class__).split(".")[-1].split("'")[0]
			num_named_children = len(list(m.named_children()))
			if num_named_children > 0:
				name = parent_name + "." + key if parent_name else key
			else:
				name = parent_name + "." + cls_name + "_" + key if parent_name else key
			names[name] = m

			if isinstance(m, torch.nn.Module):
				_get_names(m, parent_name=name)

	_get_names(model)
	return names

class Linear(nn.Module):
	"""An identity activation function"""

	def __init__(self, inplace=False):
		super(Linear, self).__init__()

	def forward(self, x):
		return x


def replace_names_dict(model):
	"""Recursive walk to get names including path."""
	names = {}

	def _get_names(module, parent_name=""):
		for key, m in module.named_children():
			cls_name = str(m.__class__).split(".")[-1].split("'")[0]
			num_named_children = len(list(m.named_children()))
			if num_named_children > 0:
				name = parent_name + "." + key if parent_name else key
				if isinstance(m, torch.nn.Module):
					_get_names(m, parent_name=name)
			else:
				# 需要拦截的函数
				name = parent_name + "." + cls_name + "_" + key if parent_name else key
				if name.find('ReLU') >= 0:
					module._modules[key] = Linear()
					name = name.replace('ReLU', 'Linear')
				elif name.find('MaxPool2d') >= 0:
					module._modules[key] = nn.AvgPool2d(module._modules[key].kernel_size, module._modules[key].stride,
														module._modules[key].padding)
					name = name.replace('MaxPool2d', 'AvgPool2d')
				names[name] = module._modules[key]

	_get_names(model)
	return names

class CalRF(object):
	def __init__(self, model):
		self._model = model
		self._summary = None
		self._hooks = None

	def _add_pre_forward(self, model):

		def add_name_forward(module, output):
			model.feature_maps = []

		return add_name_forward

	def _add_layer_forward(self, model):

		def cal_forward(module, input, output):
			model.feature_maps.append(input[0])

		return cal_forward

	def _add_model_forward(self, names_dict):
		_summary = OrderedDict()
		hooks = []
		self._summary = _summary
		self._hooks = hooks

		def register_hook(module):
			def hook(module, inputs, outputs):
				module_idx = len(_summary)
				for name, item in names_dict.items():
					if item == module:
						key = "{}_{}".format(module_idx, name)
				_summary[key] = module

			if not module._modules:
				hooks.append(module.register_forward_hook(hook))

		return register_hook

	def compute(self, input_shape, index=[-1]):
		model_handle = self._model.register_forward_pre_hook(self._add_pre_forward(self._model))
		replace_names_dict(self._model)
		# 注册模型的hook
		self._model.apply(self._add_model_forward(get_names_dict(self._model)))
		extra = torch.zeros((2, input_shape[2], input_shape[0], input_shape[1]))
		with torch.no_grad():
			self._model(extra)

		# 删除hook，防止影响
		for handle in self._hooks:
			handle.remove()
		if summaryX is None:
			raise ModuleNotFoundError('Please install torchsummaryX，'' pip install torchsummaryX ''')
		summaryX.summary(self._model, extra)

		# 添加hook
		if not isinstance(index, list):
			index = [index]
		handles = []
		print('--请不要查看FC层的感受野，因为其感受野永远是全图大小--')
		print('你查看的特征图感受野对应的名称为：')
		for i in index:
			if i < 0:
				i = len(list(self._summary.keys())) + i
			print(list(self._summary.keys())[i])
			m = self._summary.get(list(self._summary.keys())[i])
			handles.append(m.register_forward_hook(self._add_layer_forward(self._model)))

		def get_model_fn():
			self._model.eval()
			return self._model

		rf = PytorchReceptiveField(get_model_fn)
		rf_params = rf.compute(input_shape=input_shape)
		print('对应的感受野如下所示(看每一行最后一个Size即可)：')
		for maps_desc in rf.feature_maps_desc:
			size = maps_desc[1][2]
			if size[0] >= input_shape[0] and size[1] >= input_shape[1]:
				print('感受野计算可能不正确，请扩大输入图片shape，当前是{},{}'.format(input_shape[0], input_shape[1]))
			print(maps_desc)

		# 移除hook
		for handle in handles:
			handle.remove()
		model_handle.remove()
		return rf_params


def calc_receptive_filed(model, input_shape, index):
	cal_model = CalRF(model)
	cal_model.compute(input_shape, index)