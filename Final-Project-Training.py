# Final project training code
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# loading dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("nucleus", {}, "./final_dataset/trainval.json", "./final_dataset/images")

nucleus_metadata = MetadataCatalog.get("nucleus")
dataset_dicts = DatasetCatalog.get("nucleus")

# show some training examples
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=nucleus_metadata, scale=1.0)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow(str(d["file_name"]), vis.get_image()[:, :, ::-1])
    cv2.waitKey()
    cv2.destroyAllWindows()

# Training
# loading model, setting some parameters
cfg = get_cfg()
# load config file
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
# load dataset
cfg.DATASETS.TRAIN = ("nucleus",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# initialize weights from model zoo
cfg.MODEL.WEIGHTS = "model_final_2d9806.pkl"
# batch size, learning rate, iterations
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 15000
# saving model every period
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # class

# training
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
