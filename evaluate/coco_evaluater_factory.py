from evaluate.evaluate_detrac_coco_api.coco_evaluater import coco_evaluater as detrac_evaluater
from evaluate.evaluate_coco.coco_evaluater import coco_evaluater as voc_evaluater

_coco_evaluater_factory = {
	"VOC":voc_evaluater,
	"VOC_poly_yolo": voc_evaluater,
	"VOC_centernet":voc_evaluater,

	"U-DETRAC": detrac_evaluater,
	"U-DETRAC_centernet": detrac_evaluater,
}

def load_coco_evaluater(config_name):
	assert config_name in _coco_evaluater_factory, "并没有所用模型的测评类，需要添加！"
	return _coco_evaluater_factory[config_name]