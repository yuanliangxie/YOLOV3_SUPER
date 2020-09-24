import torch.nn as nn
import torch
import numpy as np
__all__ = ['label_smooth']


class label_smooth(object):
	def __init__(self, theta=0.1, classes=20):
		self.theta = theta
		self.classes = classes
	def smooth(self, tcls, mask):
		k = self.theta / (self.classes - 1)
		tcls[mask == 1] *= (1 - self.theta - k)
		tcls[mask == 1] += k
		return tcls

class mix_up(object):
	def __init__(self, alpha):
		self.alpha = alpha

	def mixup(self, images, targets):
		index = torch.randperm(images.size(0))
		lam = np.random.beta(self.alpha, self.alpha)
		output_images = lam * images + (1- lam) * images[index, :]
		targets = torch.cat((targets, targets[index, :]), 1)
		return output_images, targets

