#from models.model.poly_yolo import yolov3 as model
#from models.model.model_yolov3_baseline import yolov3 as model
from models.model.model_centernet_resnet import centernet_18 as model
#from models.model.model_LFFD import LFFD as model
from yolo_inference_test.utils_inference import *
from yolo_inference_test.visualize import visualize_boxes
#from time_analyze import print_run_time, func_line_time
os.environ["CUDA_VISIBLE_DEVICES"]='0'
class yolo_inference_detector(object):
	def __init__(self, config):
		self.config = config
		self.conf_thresh = config["CONF_THRESH"]
		self.nms_thresh = config["NMS_THRESH"]
		self.val_shape = config['TEST_IMG_SIZE']
		self.classes = config['DATA']["CLASSES"]
		self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
		self.model = model(config).to(self.device)
		self.model.eval()
		self.__load_model_weights()
		self.class_color = [(np.random.randint(255), np.random.randint(255),
							 np.random.randint(255)) for _ in range(20)]
		self.Resize = Resize(self.val_shape, correct_box=False)

	def __load_model_weights(self):
		if self.config["pretrain_snapshot"]:
			print("loading weight file from : {}".format(self.config["pretrain_snapshot"]))
			state_dict = torch.load(self.config["pretrain_snapshot"])["state_dict"]
			self.model.load_state_dict(state_dict)
		else:
			print("missing pretrain_snapshot!!!")

		print("loading weight file is done")

	#@func_line_time
	def get_bbox(self, img):
		"""

		:param img:
		:return: shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
		"""
		bboxes = self.__predict(img, self.val_shape, (0, np.inf))
		bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)
		return bboxes

	#@print_run_time
	def filter_bboxes_by_class(self, bboxes, retain_class_name=None):
		"""

		:param bboxes: shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
		:param retain_class_name: 只留下自己想要的类别
			format：list
			for example:["car"]
		:return:filter_bboxes
		"""
		########如果bboxes没有shape为(0,),则直接返回#######
		if bboxes.shape[0] == 0:
			return bboxes
		########如果没有特定可视化的类############
		if retain_class_name == None:
			return bboxes
		########取出要可视化的类##################
		class_visual_inds = []
		for name in retain_class_name:
			assert self.classes.index(name)!= None, "所要可视化的类不在所要检测的类中！"
			class_ind = self.classes.index(name)
			class_visual_inds.append(class_ind)
		class_inds = bboxes[..., 5].astype(np.int32)
		#########求取可视化的类别的mask############
		mask = np.zeros(class_inds.shape, dtype=np.bool)
		for class_visual_inds in class_visual_inds:
			sub_mask = (class_inds == class_visual_inds)
			mask = (mask | sub_mask)
		##########对bboxes进行过滤################
		bboxes = bboxes[mask, :]
		return bboxes

	#@print_run_time
	def render_bboxes(self, img, bboxes, visual_class_name = None):
		"""

		:param img: 由cv2读出的图片
		:param bboxes: bboxes为self.get_bbox返回的
		:param visual_class_name: 只显示visual_class_name里的类别，其余的不显示
		:return:
		"""
		########如果bboxes没有shape为(0,),则直接返回#######
		if bboxes.shape[0] == 0:
			return img

		########如果没有特定可视化的类############
		if visual_class_name == None:
			boxes = bboxes[..., :4]
			class_inds = bboxes[..., 5].astype(np.int32)
			scores = bboxes[..., 4]
			visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
			return img
		########取出要可视化的类##################
		class_visual_inds = []
		for name in visual_class_name:
			assert self.classes.index(name), "所要可视化的类不在所要检测的类中！"
			class_ind = self.classes.index(name)
			class_visual_inds.append(class_ind)
		class_inds = bboxes[..., 5].astype(np.int32)
		#########求取可视化的类别的mask############
		mask = np.zeros(class_inds.shape, dtype=np.bool)
		for class_visual_inds in class_visual_inds:
			sub_mask = (class_inds == class_visual_inds)
			mask = (mask | sub_mask)
		##########对bboxes进行过滤################
		bboxes = bboxes[mask, :]
		boxes = bboxes[..., :4]
		class_inds = bboxes[..., 5].astype(np.int32)
		scores = bboxes[..., 4]
		visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
		return img

	def cv2_visualize_boxes(self, image, bboxes):
		"""

		:param image:
		:param bboxes: shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class) 坐标值为绝对大小
		:return:
		"""
		for x1, y1, x2, y2, scores, class_ind in bboxes:
			cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=self.class_color[int(class_ind)], thickness=3)
			cv2.putText(image, self.classes[int(class_ind)],
						(int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
		return image

	# def cv2_render_bboxes(self, img, bboxes, visual_class_name = None):
	#     """
	#
	#     :param img: 由cv2读出的图片
	#     :param bboxes: bboxes为self.get_bbox返回的
	#     :param visual_class_name: 只显示visual_class_name里的类别，其余的不显示
	#     :return:
	#     """
	#     ########如果bboxes没有shape为(0,),则直接返回#######
	#     if bboxes.shape[0] == 0:
	#         return img
	#
	#     ########如果没有特定可视化的类############
	#     if visual_class_name == None:
	#         img = self.cv2_visualize_boxes(img, bboxes)
	#         return img
	#     ########取出要可视化的类##################
	#     class_visual_inds = []
	#     for name in visual_class_name:
	#         assert self.classes.index(name), "所要可视化的类不在所要检测的类中！"
	#         class_ind = self.classes.index(name)
	#         class_visual_inds.append(class_ind)
	#     class_inds = bboxes[..., 5].astype(np.int32)
	#     #########求取可视化的类别的mask############
	#     mask = np.zeros(class_inds.shape, dtype=np.bool)
	#     for class_visual_inds in class_visual_inds:
	#         sub_mask = (class_inds == class_visual_inds)
	#         mask = (mask | sub_mask)
	#     ##########对bboxes进行过滤################
	#     bboxes = bboxes[mask, :]
	#     img = self.cv2_visualize_boxes(img, bboxes)
	#     return img

	#@func_line_time
	def __predict(self, img, test_shape, valid_scale):
		#org_img = np.copy(img)
		org_h, org_w, _ = img.shape

		img = self.__get_img_tensor(img, test_shape).to(self.device)
		#self.model.eval()
		with torch.no_grad():
			#torch.cuda.synchronize()
			#t0 = time.time()
			outputs = self.model(img)
			#torch.cuda.synchronize()
			#t1 = time.time()
			#print("model_inference_time%f"%(t1-t0))
			p_d = torch.cat(outputs, 1)
		pred_bbox = p_d.squeeze().cpu().numpy()
		bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)
		return bboxes

	#@func_line_time
	def __get_img_tensor(self, img, test_shape):

		img = self.Resize(img, None)
		img = img.transpose(2, 0, 1)
		return torch.from_numpy(img[np.newaxis, ...]).type(torch.FloatTensor)

	#@func_line_time
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
		infer_h, infer_w = test_input_size
		resize_ratio = min(1.0 * infer_w / org_w, 1.0 * infer_h / org_h)
		dw = (infer_w - resize_ratio * org_w) / 2
		dh = (infer_h - resize_ratio * org_h) / 2
		pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
		pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

		#(1)将score低于score_threshold的bbox去掉
		classes = np.argmax(pred_prob, axis=-1)
		scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
		score_mask = scores > self.conf_thresh
		pred_coor = pred_coor[score_mask]
		scores = scores[score_mask]
		classes = classes[score_mask]

		# (2)将预测的bbox中超出原图的部分裁掉
		pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
									np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
		# (3)将无效bbox的coor置为0
		invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
		valide_mask = ~invalid_mask
		#pred_coor[invalid_mask] = 0

		# (4)去掉不在有效范围内的bbox
		bboxes_scale = np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1)
		scale_mask = np.logical_and((valid_scale[0]**2 < bboxes_scale), (bboxes_scale < valid_scale[1]**2))

		mask = np.logical_and(scale_mask, valide_mask)
		coors = pred_coor[mask]
		scores = scores[mask]
		classes = classes[mask]

		bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

		return bboxes

if __name__ == '__main__':
	import time
	from config.inference_config_U_Detrac import TEST as config
	detector = yolo_inference_detector(config)
	img = cv2.imread('./pic/2017.01.10 11.50.09_06.jpg')
	t0 = time.time()
	bboxes_prd = detector.get_bbox(img)
	t1 = time.time()
	print("inference_time:%.5f"%(t1-t0))
	render_img = detector.render_bboxes(img, bboxes_prd)
	cv2.imshow('img', render_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



