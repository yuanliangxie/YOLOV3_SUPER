from .base import ReceptiveField
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Callable, List
from .types import (
	ImageShape,
	GridPoint,
	GridShape,
	FeatureMapDescription,
)

def _define_receptive_field_func(
		model_fn: Callable[[], nn.Module], input_shape: GridShape
):
	shape = [input_shape.n, input_shape.c, input_shape.h, input_shape.w]
	input_image = torch.zeros(*shape)
	model = model_fn()
	_ = model(input_image)

	feature_maps = model.__dict__.get('feature_maps', None)
	if feature_maps is None:
		raise ValueError('Field module.feature_maps is not defined. '
						 'Cannot compute receptive fields.')

	if type(feature_maps) != list:
		raise ValueError('Field module.feature_maps must be a list. '
						 'Cannot compute receptive fields.')

	# compute feature maps output shapes
	output_shapes = []
	for feature_map in feature_maps:
		output_shape = feature_map.size()
		output_shape = GridShape(
			n=output_shape[0],
			c=output_shape[1],
			h=output_shape[2],
			w=output_shape[3]
		)
		output_shapes.append(output_shape)

	print(f"Feature maps shape: {output_shapes}")
	print(f"Input shape       : {input_shape}")

	def gradient_function(
			receptive_field_masks: List[torch.Tensor]
	) -> List[np.ndarray]:

		grads = []
		for fm, rf_mask in enumerate(receptive_field_masks):
			input_tensor = torch.zeros(*shape)
			input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
			model.zero_grad()
			_ = model(input_tensor)

			fm = torch.mean(model.feature_maps[fm], 1, keepdim=True)
			fake_loss = fm * rf_mask
			fake_loss = torch.mean(fake_loss)
			fake_loss.backward()
			grads.append(input_tensor.grad.detach().numpy())

		return grads

	return gradient_function, input_shape, output_shapes



class PytorchReceptiveField(ReceptiveField):
	def __init__(self, model_func: Callable[[], nn.Module]):
		"""
		:param model_func: model function which returns new instance of
			nn.Module.
		"""
		super().__init__(model_func)

	def _prepare_gradient_func(
			self, input_shape: ImageShape
	) -> Tuple[Callable, GridShape, List[GridShape]]:
		"""
		Computes gradient function and additional parameters. Note
		that the receptive field parameters like stride or size, do not
		depend on input image shape. However, if the RF of original network
		is bigger than input_shape this method will fail. Hence it is
		recommended to increase the input shape.
		:param input_shape: shape of the input image, which is feed to the
			Pytorch module.
		:returns
			gradient_function: a function which returns gradient w.r.t. to
				the input image.
			input_shape: a shape of the input image tensor.
			output_shapes: a list shapes of the output feature map tensors.
		"""
		input_shape = ImageShape(*input_shape)
		input_shape = GridShape(
			n=1,
			c=input_shape.c,
			h=input_shape.h,
			w=input_shape.w,
		)

		gradient_function, input_shape, output_shapes = \
			_define_receptive_field_func(self._model_func, input_shape)

		return gradient_function, input_shape, output_shapes

	def _get_gradient_from_grid_points(
			self, points: List[GridPoint], intensity: float = 1.0
	) -> List[np.ndarray]:
		"""
		Computes gradient at image tensor generated by
		point-like perturbation at output grid location given by
		@point coordinates.
		:param points: source coordinate of the backpropagated gradient for each
			feature map.
		:param intensity: scale of the gradient, default = 1
		:return a list gradient maps of shape [1, C, H, W] for each feature map
		"""
		output_feature_maps = []
		for fm in range(self.num_feature_maps):
			os = self._output_shapes[fm]
			# input tensors are in NCHW format
			output_feature_map = torch.zeros(size=[1, 1, os.h, os.w])
			output_feature_map[:, :, points[fm].y, points[fm].x] = intensity
			output_feature_maps.append(output_feature_map)

		def _postprocess_grad(grad: np.ndarray) -> np.ndarray:
			# convert to NHWC format
			grad = np.abs(np.transpose(grad, [0, 2, 3, 1]))
			return grad / grad.max()

		torch_grads = self._gradient_function(output_feature_maps)
		return [_postprocess_grad(grad) for grad in torch_grads]

	def compute(self, input_shape: ImageShape) -> List[FeatureMapDescription]:
		"""
		Compute ReceptiveFieldDescription of given model for image of
		shape input_shape [H, W, C]. If receptive field of the network
		is bigger thant input_shape this method will raise exception.
		In order to solve this problem try to increase input_shape.
		:param input_shape: shape of the input image e.g. (224, 224, 3)
		:return a list of estimated FeatureMapDescription for each feature
			map.
		"""
		return super().compute(input_shape=input_shape)