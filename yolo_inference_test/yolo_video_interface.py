from yolo_inference_test.inference_pipline import yolo_inference_detector
from yolo_inference_test.utils_inference import video_save_from_capture, xyxy2xywh
import numpy as np
#from time_analyze import print_run_time, func_line_time
import time
import cv2

class vehicle_detector(yolo_inference_detector):
	def __init__(self, config):
		"""

		:param config: 参数设置
		:param video_name: 进行处理的视频的名字
		:param video_imgsize: 保存视频的图像大小
		:param video_save_fps: 保存的视频帧率
		"""
		super().__init__(config)

	def deploy_inference_video(self, video_name, video_save_name, video_imgsize=None, video_save_fps=30 ):
		self.cap = cv2.VideoCapture(video_name)
		if video_imgsize:
			self.video_imgsize = video_imgsize
		else:
			self.video_imgsize = self.set_origin_video_imgsize()

		self.video_saver = video_save_from_capture(video_save_name, self.video_imgsize, video_save_fps)

	def set_origin_video_imgsize(self):
		"""

		:return: (w,h)
		"""
		rec, frame = self.cap.read()
		if rec:
			imgsize = frame.shape
		return imgsize[:-1][::-1]

	def cv2_track_visualize_boxes(self, image, bboxes):
		for x1, y1, x2, y2, identities, class_ind in bboxes:
			cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=self.class_color[int(identities)%20], thickness=3)
			cv2.putText(image, "ID:%d"%int(identities),
						(int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
		return image

	#@func_line_time
	def inference_image(self, image):
		"""

		:param image: 输入正常的由cv2读取的图片
		:return: 经过检测模型并可视化了检测框
		"""
		bboxes = self.get_bbox(image)
		bboxes = self.filter_bboxes_by_class(bboxes)
		inference_image = self.cv2_visualize_boxes(image, bboxes)
		return inference_image

	def cal_inference_time(self):
		"""
		纯计算检测速度，不进行显示
		"""
		count = 0
		total_time = 0
		while(True):
			rec, frame = self.cap.read()
			if not rec:
				break
			start_time = time.time()
			self.get_bbox(frame)
			end_time = time.time()
			count += 1
			if count > 200:
				total_time += (end_time - start_time)
				average_time = total_time / (count-200)
				fps = 1 / average_time
				print("FPS:%.2f"%fps)
		print("average_fps:%.2f"%fps)
		self.cap.release()

	def show_video(self,):
		"""
		对结果进行了显示，且计算了fps
		"""
		count = 0
		total_time = 0
		while(True):
			start_time = time.time()

			rec, frame = self.cap.read()
			if not rec:
				break
			frame = cv2.resize(frame, self.video_imgsize)
			image = self.inference_image(frame)
			#image = self.track_image(frame)

			end_time = time.time()

			total_time += (end_time - start_time)
			count += 1
			average_time = total_time / count
			fps = 1 / average_time
			cv2.putText(image, "FPS:%.2f"%fps,
						(40, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 255), 1)
			cv2.imshow('vehicle', image)

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

			self.video_saver.write(image)
		print("average_fps:%.2f"%fps)
		self.cap.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	#from yolo_inference_test.poly_yolo_config_voc import TEST as config
	#from yolo_inference_test.centernet_config_detrac import TEST as config
	#from yolo_inference_test.yolov3_config_voc import TEST as config
	#from yolo_inference_test.LFFD_config_detrac import TEST as config
	#from yolo_inference_test.tiny_yolov3_config_detrac import TEST as config
	#from yolo_inference_test.yolov3_config_detrac import TEST as config
	#from yolo_inference_test.LVnet_iou_assign import TEST as config
	#from yolo_inference_test.LVnet_no_fpn_centerloss_yololoss import TEST as config
	#from yolo_inference_test.mobilenetv2_yolov3_config_detrac import TEST as config
	from yolo_inference_test.shulffnetv2_yolov3_config_detrac import TEST as config
	#from yolo_inference_test.LVnet_largest_fpn_config_detrac import TEST as config
	detector = vehicle_detector(config)
	#配置输入视频
	detector.deploy_inference_video(video_name='./Radar1150_30.avi', video_save_name='LVnet_largest_fpn_detrac_detect.avi', video_save_fps=30)
	#detector.cal_inference_time()
	detector.show_video()