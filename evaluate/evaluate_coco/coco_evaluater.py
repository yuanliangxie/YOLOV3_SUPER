from evaluate.evaluate_coco.evaluater_proto import Evaluator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import numpy as np
import json
current_file_path = os.path.dirname(os.path.abspath(__file__))
class coco_evaluater(Evaluator):
    def __init__(self, model, config, visiual=True):
        super().__init__(model, config, visiual)
        self.gt_save_path = os.path.join(current_file_path, "voc2007_test.json")
        self.pred_save_path = os.path.join(current_file_path, "pred_result.json")
        self.generate_eval_result_json()

    def generate_eval_result_json(self):
        data_dict = []
        for img_ind, bboxes_pred in self.yiled_result_bboxes_prd():
            for bbox in bboxes_pred:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                score = float('%.4f' % score)
                class_ind = int(bbox[5])
                xmin, ymin, xmax, ymax = map(float, coor)
                bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
                A = {"image_id":img_ind, "category_id": class_ind, "bbox": bbox,
                     "score": score}  # COCO json format
                data_dict.append(A)
        if len(data_dict) > 0:
            print('evaluating ......')
            json.dump(data_dict, open(self.pred_save_path, 'w'))

    def eval(self):
        cocoGt = COCO(self.gt_save_path)  # 标注文件的路径及文件名，json文件形式
        cocoDt = cocoGt.loadRes(self.pred_save_path)  # 自己的生成的结果的路径及文件名，json文件形式
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        state = cocoEval.summarize()
        if self.config["generate_analyze_figure"]:
            cocoEval.analyze('./'+self.config['generate_analyze_figure_dir_name'])
        return state





