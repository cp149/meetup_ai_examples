# Some basic setup:
# Setup detectron2 logger
import detectron2
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import os
from detectron2.data import MetadataCatalog


def detect_img(predictor, im):
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    v = Visualizer(
        im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("./detectron2-result.jpg", out.get_image()[:, :, ::-1])
    cv2.waitKey()


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
base_dir = "./car"
predictor = DefaultPredictor(cfg)
for test_file in os.listdir(base_dir):
    if test_file.endswith(".gif"):
        cap = cv2.VideoCapture(os.path.join(base_dir, test_file))
        ret, image = cap.read()
        cap.release()
        if cap:
            im = image
    else:
        im = cv2.imread(os.path.join(base_dir, test_file))
    if im is not None:
        detect_img(predictor, im)
