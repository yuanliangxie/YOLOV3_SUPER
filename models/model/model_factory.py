from models.model.model_yolov3_baseline import yolov3 as yolov3
from models.model.model_yolov3_mobilev2 import yolov3_mobilev2 as yolov3_mobile
from models.model.model_yolov3_x import yolov3 as yolov3_x
from models.model.poly_yolo import yolov3 as poly_yolo

_model_factory = {
	"yolov3": yolov3,
	"yolov3_mobile": yolov3_mobile,
	"yolov3_x": yolov3_x,
	"poly_yolo": poly_yolo
}

def load_model(model_name):
	assert model_name in _model_factory, '你想要加载的模型不在工厂模型中！'
	return _model_factory[model_name]