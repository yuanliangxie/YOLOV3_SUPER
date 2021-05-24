
from typing import Tuple, List

import numpy as np

from .types import GridShape, ReceptiveFieldRect
def estimate_rf_from_gradient(receptive_field_grad: np.ndarray) -> ReceptiveFieldRect:
	"""
	Given input gradient tensors of shape [N, W, H, C] it returns the
	estimated size of gradient `blob` in W-H directions i.e. this
	function computes the size of gradient in W-H axis for each feature map.
	:param receptive_field_grad: a numpy tensor with gradient values
		obtained for certain feature map
	:return: a corresponding ReceptiveFieldRect
	"""

	receptive_field_grad = np.array(receptive_field_grad).mean(0).mean(-1)
	binary_map: np.ndarray = (receptive_field_grad[:, :] > 0)

	x_cs: np.ndarray = binary_map.sum(-1) >= 1
	y_cs: np.ndarray = binary_map.sum(0) >= 1

	x = np.arange(len(x_cs))
	y = np.arange(len(y_cs))

	width = x_cs.sum()
	height = y_cs.sum()

	x = np.sum(x * x_cs) / width
	y = np.sum(y * y_cs) / height

	return ReceptiveFieldRect(x, y, width, height)


def estimate_rf_from_gradients(
		receptive_field_grads: List[np.ndarray]
) -> List[ReceptiveFieldRect]:
	"""
	Given input gradient tensors of shape [N, W, H, C] it returns the
	estimated size of gradient `blob` in W-H directions i.e. this
	function computes the size of gradient in W-H axis for each feature map.
	:param receptive_field_grads: a list of numpy tensor with gradient values
		obtained for different feature maps
	:return: a list of corresponding ReceptiveFieldRect
	"""

	return [
		estimate_rf_from_gradient(receptive_field_grad)
		for receptive_field_grad in receptive_field_grads
	]