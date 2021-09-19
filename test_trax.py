# Some basic setup:
# Setup detectron2 logger
import detectron2
import cv2
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

if __name__ == "__main__":
    model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    register_coco_instances("trax", {}, "./mask/test_coco.json", "./")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.DATASETS.TRAIN = ("trax",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = os.path.join("./output", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=["Trax"])
    predictor = DefaultPredictor(cfg)
    for x in os.listdir("./test"):
        im = cv2.imread(os.path.join("./test", x))
        outputs = predictor(im)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        v = Visualizer(
            im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        show_img = out.get_image()[:, :, ::-1]
        cv2.imshow("./detectron2-result.jpg", show_img)
        cv2.waitKey()
