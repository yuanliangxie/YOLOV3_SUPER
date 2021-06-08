import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
from dataset import ssd_augmentations_ATR_data
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False



class ATR_sky_Dataset(Dataset):
	def __init__(self, list_path, labels_path, img_size, is_training, is_debug=False, batch_size=16):
		self.img_files = []
		self.label_files = []
		self.batch_size = batch_size
		# if is_training:
		#     labels_path = '../data/voc/labels'
		# else:
		#     labels_path = '../data/voc/labels_test'
		for path in open(list_path, 'r'): #'vehecal/labels'
			path_split = path.split('/')
			index_anno = path_split[-2]
			index_bmp = path_split[-1]
			label_path = os.path.join(labels_path, index_anno+'_'+index_bmp.replace('.bmp', '.txt')).strip()
			# label_path = path.replace('VOC2007/JPEGImages', 'labels').replace('VOC2012/JPEGImages', 'labels').replace(
			#  'VOC2007_test/JPEGImages', 'labels_test').replace('.jpg', '.txt').strip()
			if os.path.isfile(label_path):
				self.img_files.append(path)
				self.label_files.append(label_path)
			else:
				print("no label found. skip it: {}".format(path))
		#print("Total images: {}".format(len(self.img_files)))
		#self.img_size = img_size  # (w, h)
		#self.max_objects = 50#这里的max_object是最大的bbox的数目
		self.is_debug = is_debug

		#  transforms and augmentation
		self.total_transforms = ssd_augmentations_ATR_data.Compose(transforms=[])

		# self.transforms.add(data_transforms.KeepAspect())

		if is_training:
			self.total_transforms.add(ssd_augmentations_ATR_data.SSDAugmentation())
			if img_size is None:
				self.total_transforms.add(ssd_augmentations_ATR_data.ResizeImage_multi_scale(batch_size=self.batch_size))
			else:
				self.total_transforms.add(ssd_augmentations_ATR_data.ResizeImage_single_scale(new_size=img_size))
		else:
			self.total_transforms.add(ssd_augmentations_ATR_data.ToAbsoluteCoords())
			self.total_transforms.add(ssd_augmentations_ATR_data.xywh_to_xyxy())
			self.total_transforms.add(ssd_augmentations_ATR_data.ResizeImage_single_scale(new_size=img_size))
		self.total_transforms.add(ssd_augmentations_ATR_data.xyxy_to_xywh())
		self.total_transforms.add(ssd_augmentations_ATR_data.ToPercentCoords())
		self.total_transforms.add(ssd_augmentations_ATR_data.ToTensor(self.is_debug))

	def index_2_classes(self, classes):
		index = [i for i in range(len(classes))]
		index_2_class = dict(zip(index, classes))
		return index_2_class
	def classes_2_index(self, classes):
		index = [i for i in range(len(classes))]
		class_2_index = dict(zip(classes, index))
		return class_2_index

	def __getitem__(self, index):
		#print('index='+str(index))
		img_path = self.img_files[index].rstrip()
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		if img is None:
			raise Exception("Read image error: {}".format(img_path))
		ori_h, ori_w = img.shape[:2]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#TODO:加入训练的数据是RGB格式的

		label_path = self.label_files[index].rstrip()
		if os.path.exists(label_path):
			labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)#np.loadtxt可以直接加载文本中具有固定格式的文字数据
		else:
			print("label does not exist: {}".format(label_path))
			labels = np.zeros((1, 5), np.float32)
		image = img
		box = labels[:, 1:]
		label = labels[:, 0].reshape(-1, 1)

		if self.total_transforms is not None:
			image, box, label = self.total_transforms(image, box, label)

		label = torch.cat((label, box), 1)
		sub_list = img_path.split('.')[0].split('/')
		img_ind = sub_list[-2] + '_' + sub_list[-1]
		#img_ind = img_path[-10:-4]
		# sample = {'image': image, 'label': torch.cat((label, box),1)}
		# sample["image_path"] = img_path[-10:-4]
		origin_size = [ori_h, ori_w]
		return image, label, img_ind, origin_size

	def __len__(self):
		return len(self.img_files)

	def collate_fn(self, batch):
		'''Pad images and encode targets.

			As for images are of different sizes, we need to pad them to the same size.

			Args:
			  batch: (list) of images, cls_targets, loc_targets.

			Returns:
			  padded images, stacked cls_targets, stacked loc_targets.

			Reference:
			  https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/blob.py
			'''
		batchsize = len(batch)
		image = [x[0] for x in batch]
		label = [x[1] for x in batch]
		img_ind = [x[2] for x in batch]
		origin_size = [x[3] for x in batch]
		max_objects = max([x.shape[0] for x in label])
		label_new = []
		for i in range(batchsize):
			label_len = label[i].shape[0]
			filled_labels = torch.zeros(max_objects, 5).type(torch.FloatTensor)
			filled_labels[range(max_objects)[:label_len]] = label[i][:]
			label_new.append(filled_labels)
		sample = {'image': torch.stack(image), 'label': torch.stack(label_new), 'img_ind': img_ind, "origin_size": origin_size}
		return sample

class Statistic_scale:
	def __init__(self, continus_assign_scale, size_scale, index="_"):
		"""
		:param continus_assign_scale:
		:param size_scale:
		:param index: 代表所要保存的图片的特征标识
		"""
		self.index = index
		self.assign_scales = continus_assign_scale
		self.assign_area_scales = [[assign_scale[0]**2, assign_scale[1]**2] for assign_scale in self.assign_scales]
		self.size_scale= size_scale

		self.scaled_area_data_x = np.arange(1, len(continus_assign_scale)+1+1)
		self.scaled_max_length_data_x = np.arange(1, len(continus_assign_scale)+1+1)
		self.max_length_data_x = np.arange(1, 51)
		self.statistics_scaled_area_data = np.zeros(len(continus_assign_scale)+1)
		self.statistics_max_length_data = np.zeros(50)
		self.statistics_scaled_max_length_data = np.zeros(len(continus_assign_scale)+1)
		self.boxes_numbers = 0
		self.collection_unassign_length = []


	def statistics_max_length(self, max_length):
		max_length = int(max_length)
		if max_length < 1:
			pass
		else:
			if max_length == 640:
				max_length = max_length-1
			self.statistics_max_length_data[max_length] += 1

	def statistics_scaled_area(self, area):
		assign = False
		for i, assign_area_scale in enumerate(self.assign_area_scales):
			if assign_area_scale[0] <= area < assign_area_scale[1]:
				self.statistics_scaled_area_data[i] += 1
				assign = True
				break
		if assign == False:
			self.statistics_scaled_area_data[i] += 1
		pass

	def statistics_scaled_max_length(self, max_length):
		assign = False
		for i, assign_scale in enumerate(self.assign_scales):
			if assign_scale[0] <= max_length < assign_scale[1]:
				self.statistics_scaled_max_length_data[i] += 1
				assign = True
				break
		if assign == False:
			self.statistics_scaled_max_length_data[-1] += 1
			self.collection_unassign_length.append(max_length)

	def run(self, max_length, area):
		self.statistics_scaled_area(area)
		self.statistics_max_length(max_length)
		self.statistics_scaled_max_length(max_length)

	def only_run_max_length(self, max_length):
		self.statistics_max_length(max_length)

	def only_show_max_length(self):
		plt.figure(1, figsize=[5, 3])
		ax1 = plt.subplot(1,1,1)
		plt.sca(ax1)
		ax1.xaxis.set_major_locator(plt.MultipleLocator(5))
		plt.bar(self.max_length_data_x, self.statistics_max_length_data, align="center", color="c")
		plt.xlabel("Distribution of the longest side")
		plt.ylabel("Amount")
		plt.tight_layout()
		plt.savefig("./Distribution_of_the_longest_side"+self.index+".png", dpi=300)
		plt.show()
		plt.close()





	def show(self):
		self.show_percent()

		plt.figure(1, figsize=[9, 3])
		ax1 = plt.subplot(1, 3, 1)
		ax2 = plt.subplot(1, 3, 2)
		ax3 = plt.subplot(1, 3, 3)


		plt.sca(ax1)
		ax1.xaxis.set_major_locator(plt.MultipleLocator(128))
		plt.bar(self.max_length_data_x, self.statistics_max_length_data, align="center", color="c")
		plt.xlabel("Distribution of the longest side")
		plt.ylabel("Amount")


		plt.sca(ax2)
		ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
		plt.bar(self.scaled_area_data_x, self.statistics_scaled_area_data, align="center", color="c")
		plt.xlabel("Area distribution in multiple ranges")
		plt.ylabel("Amount")



		plt.sca(ax3)
		ax3.xaxis.set_major_locator(plt.MultipleLocator(1))
		plt.bar(self.scaled_max_length_data_x, self.statistics_scaled_max_length_data, align="center", color="c")
		plt.xlabel("Long-side distribution in multiple ranges")
		plt.ylabel("Amount")


		plt.tight_layout()

		plt.savefig("./assign_scale_result"+self.index+".png", dpi=300)
		#plt.show()
		plt.close()

	def show_percent(self):
		print(list(self.statistics_scaled_area_data/np.sum(self.statistics_scaled_area_data)))


#  use for test dataloader
if __name__ == "__main__":
	import sys
	from tqdm import tqdm
	import train.ATR_data_preprocess.params_init_ATR as params_init
	MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
	sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))


	classes_category = params_init.TRAINING_PARAMS["model"]["classes_category"]
	dataset = ATR_sky_Dataset("../data/ATR_sky/trainval.txt", "../data/ATR_sky/labels",
							  (640, 640), False, is_debug=True, batch_size=8)
	index_2_classes = dataset.index_2_classes(classes_category)
	dataloader = torch.utils.data.DataLoader(dataset,
											 batch_size=8,
											 shuffle=False, num_workers=0, pin_memory=False, collate_fn=dataset.collate_fn)
	print(len(dataset))

	size_scale = 640
	continue_assign_scale = [[1, 20], [20, 100], [100, 260]]
	statisticser = Statistic_scale(continue_assign_scale, size_scale, index="ATR_data_statistic_result")
	for step, sample in tqdm(enumerate(dataloader)):
		for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
			image = image.numpy()
			label = label.numpy()
			h, w = image.shape[:2]
			for l in label:
				if l.sum() == 0:
					continue
				gt_h = l[4] * h
				gt_w = l[3] * w

				area = gt_h * gt_w
				max_length = min(gt_h, gt_w)

				#statisticser.run(max_length, area)
				statisticser.only_run_max_length(max_length)
	statisticser.only_show_max_length()
	print(statisticser.collection_unassign_length)
