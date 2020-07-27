from models.model.model_yolov3_baseline import yolov3 as yolov3
from models.model.model_yolov3_mobilev2 import yolov3_mobilev2 as yolov3_mobile
from models.model.model_yolov3_x import yolov3 as yolov3_x

_model_factory = {
	"yolov3": yolov3,
	"yolov3_mobile": yolov3_mobile,
	"yolov3_x": yolov3_x,
}

def load_model(model_name):
	return _model_factory[model_name]