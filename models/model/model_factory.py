from models.model.model_yolov3_baseline import yolov3 as yolov3
from models.model.model_yolov3_mobilev2 import yolov3_mobilev2 as yolov3_mobile
from models.model.model_yolov3_x import yolov3 as yolov3_x
from models.model.poly_yolo import yolov3 as poly_yolo
from models.model.model_centernet_resnet import centernet_18 as centernet_18
from models.model.model_LFFD import LFFD
from models.model.model_LVnet import LVnet
from models.model.model_LVnet_with_deconv import LVnet as LVnet_with_deconv
from models.model.model_LVnet_with_deconv_shallow_centerloss import LVnet as LVnet_with_deconv_centerloss
from models.model.model_LVnet_with_pure_centerloss import LVnet as LVnet_with_pure_centerloss
from models.model.model_LVnet_with_deconv_shalllow_centerloss_iou_assign import LVnet as LVnet_with_iou_assign
from models.model.model_LVnet_oringin_iou_assign_no_fpn import LVnet as LVnet_with_iou_assign_no_fpn
from models.model.model_tiny_yolov3 import tiny_yolov3 as tiny_yolov3
_model_factory = {
	"yolov3": yolov3,
	"yolov3_mobile": yolov3_mobile,
	"yolov3_x": yolov3_x,
	"poly_yolo": poly_yolo,
	"centernet_18": centernet_18,
	"LFFD": LFFD,
	"LVnet":LVnet,
	"LVnet_with_deconv":LVnet_with_deconv,
	"LVnet_with_deconv_centerloss": LVnet_with_deconv_centerloss,
	"LVnet_with_pure_centerloss":LVnet_with_pure_centerloss,
	"LVnet_with_iou_assign":LVnet_with_iou_assign,
	"LVnet_with_iou_assign_no_fpn":LVnet_with_iou_assign_no_fpn,
	"tiny_yolov3":tiny_yolov3,
}

def load_model(model_name):
	assert model_name in _model_factory, '你想要加载的模型不在工厂模型中！'
	return _model_factory[model_name]