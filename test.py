from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from src.lstm import ActionClassificationLSTM

import cv2


# obtain detectron2's default config
cfg = get_cfg()
# load the pre trained model from Detectron2 model zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
# set confidence threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# load model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
# create the predictor for pose estimation using the config
pose_detector = DefaultPredictor(cfg)


img = cv2.imread('team_photo.png')

poses = pose_detector(img)

#this outputs the keypoints in format [x,y,confidence]
print(poses['instances'].pred_keypoints)

lstm_classifier = ActionClassificationLSTM.load_from_checkpoint("models/saved_model.ckpt")
print(type(lstm_classifier))