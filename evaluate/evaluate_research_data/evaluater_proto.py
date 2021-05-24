import os
import shutil
from tqdm import tqdm
from evaluate.evaluate_research_data.utils_voc import *
from evaluate.evaluate_research_data.visualize import visualize_boxes
from dataset.research_dataset_add_SSD_aug import Research_Dataset as Dataset


class Evaluator(object):
	def __init__(self, model, config, visiual=True):
		self.config = config
		self.classes = config['DATA']["CLASSES"]
		self.pred_result_path = os.path.join(config['PROJECT_PATH'], 'evaluate/evaluate_research_data/data', 'results')
		self.conf_thresh = config["CONF_THRESH"]
		self.nms_thresh = config["NMS_THRESH"]
		self.val_shape = config["TEST_IMG_SIZE"]
		self.__visiual = visiual
		self.__visual_imgs = 0

		self.model = model
		self.device = next(model.parameters()).device
		assert len(config["test_path"]) != 0, "请确保config中的test_path不为None！"
		self.is_exists_pred_result_path()
		self.generator = self.initial_generator()

	def is_exists_pred_result_path(self):
		if os.path.exists(self.pred_result_path):
			shutil.rmtree(self.pred_result_path)
		os.makedirs(self.pred_result_path)

	def initial_generator(self):
		dataset = Dataset(list_path=self.config["test_path"], labels_path=self.config["test_labels_path"], img_size=(self.val_shape, self.val_shape),
							 is_training=False, is_debug=False, batch_size=16)
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config['BATCH_SIZE'],
												 shuffle=False, num_workers=16, pin_memory=True,
												 collate_fn=dataset.collate_fn)
		return dataloader

	def yiled_result_bboxes_prd(self, multi_test=False, flip_test=False):
		for _, sample in enumerate(tqdm(self.generator)):
			for index, (img, img_ind, orgin_shape) in enumerate(zip(sample['image'], sample['img_ind'], sample['origin_size'])):
				img = img.unsqueeze(0)
				bboxes = self.__predict(img, self.val_shape, orgin_shape, (0, np.inf))



				bboxes[..., 5] = 0  #将所有的标签改为0，先测试要紧



				bboxes_prd = nms(bboxes, self.conf_thresh, self.nms_thresh)
				if bboxes_prd.shape[0]!=0 and self.__visiual and self.__visual_imgs % (len(self.generator)*self.config['BATCH_SIZE']//100) == 0:
					boxes = bboxes_prd[..., :4]
					class_inds = bboxes_prd[..., 5].astype(np.int32)
					scores = bboxes_prd[..., 4]
					img = img[0, ...].numpy().transpose(1, 2, 0)
					img = (img * 255).astype(np.uint8)
					img = self.__convert_image(img, orgin_shape)
					img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
					visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
					path = os.path.join(self.pred_result_path, "{}.jpg".format(self.__visual_imgs))
					cv2.imwrite(path, img)
				self.__visual_imgs += 1
				yield img_ind, bboxes_prd

	def get_bbox(self, img, multi_test=False, flip_test=False):
		if multi_test:
			test_input_sizes = range(320, 640, 96)
			bboxes_list = []
			for test_input_size in test_input_sizes:
				valid_scale =(0, np.inf)
				bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
				if flip_test:
					bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
					bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
					bboxes_list.append(bboxes_flip)
			bboxes = np.row_stack(bboxes_list)
		else:
			bboxes = self.__predict(img, self.val_shape, (0, np.inf))

		bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)

		return bboxes

	def __predict(self, img, test_shape, org_image_shape, valid_scale):
		"""

		:param img:
		:param test_shape:
		:param org_image_shape: (org_h, org_w)
		:param valid_scale:
		:return:
		"""
		self.model.eval()
		with torch.no_grad():
			outputs = self.model(img.to(self.device))
			p_d = torch.cat(outputs, 1)
		pred_bbox = p_d.squeeze().cpu().numpy()
		bboxes = self.__convert_pred(pred_bbox, test_shape, org_image_shape, valid_scale)
		return bboxes

	def __get_img_tensor(self, img, test_shape):
		img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
		return torch.from_numpy(img[np.newaxis, ...]).float()


	def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
		"""
		预测框进行过滤，去除尺度不合理的框
		"""
		pred_coor = xywh2xyxy(pred_bbox[:, :4])
		pred_conf = pred_bbox[:, 4]
		pred_prob = pred_bbox[:, 5:]

		# (1)
		# (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
		# (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
		# 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
		# 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
		org_h, org_w = org_img_shape
		resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
		dw = (test_input_size - resize_ratio * org_w) // 2
		dh = (test_input_size - resize_ratio * org_h) // 2
		pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
		pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

		# (2)将预测的bbox中超出原图的部分裁掉
		pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
									np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
		# (3)将无效bbox的coor置为0
		invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
		pred_coor[invalid_mask] = 0

		# (4)去掉不在有效范围内的bbox
		bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
		scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

		# (5)将score低于score_threshold的bbox去掉
		classes = np.argmax(pred_prob, axis=-1)
		scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
		score_mask = scores > self.conf_thresh

		mask = np.logical_and(scale_mask, score_mask)

		coors = pred_coor[mask]
		scores = scores[mask]
		classes = classes[mask]

		bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

		return bboxes

	def __convert_image(self, image, oringin_size):
		"""
		使图片固定aspect ratio，并且更新boxes
		:param image:image_plus
		:param boxes: shape = [:,4], format:xyxy
		:param expected_size:(,)
		:return:
		"""
		ih, iw = oringin_size
		eh, ew, _ = image.shape
		scale = min(eh / ih, ew / iw)
		nh = int(ih * scale)
		nw = int(iw * scale)
		top = (eh - nh) // 2
		left = (ew - nw) // 2
		image_input_size = image[top: top+nh, left:left+nw, :]
		image_input_size = cv2.resize(image_input_size, (iw, ih))
		return image_input_size



