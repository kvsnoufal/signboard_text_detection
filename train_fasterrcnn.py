import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle as cp

from PIL import ImageFont, ImageDraw, Image
import glob
import os
from tqdm import tqdm
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
import config

def get_mask(fname):
    # print(fname)
    mask = cv2.imread(fname)
    im_bw = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(im_bw,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)[-2:]
    x,y,w,h = cv2.boundingRect(contours[0])
    return [x,y,x+w,y+h]

def get_annotation_dict_from_mask(ims):

    dataset = []
    for idx,filename in enumerate(ims):
        record = {}
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []
        masks = {}
        if "ar_"+os.path.basename(filename) in ar_masknames:
            masks[0] = os.path.join("./imagefolder/genmasks","ar_"+os.path.basename(filename))
        if "eng_"+os.path.basename(filename) in eng_masknames:
            masks[1] = os.path.join("./imagefolder/genmasks","eng_"+os.path.basename(filename))
        for key,mask_filename in masks.items():
            obj = {
                    "bbox": get_mask(mask_filename),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "segmentation": [poly],
                    "category_id": key,
                }
            objs.append(obj)

        record["annotations"] = objs
        dataset.append(record)
    return dataset

images = glob.glob("./imagefolder/genimages/*.jpg")
all_masknames = glob.glob("./imagefolder/genmasks/*.jpg")
ar_masknames = [os.path.basename(fname) for fname in all_masknames if "ar_" in fname]
eng_masknames = [os.path.basename(fname) for fname in all_masknames if "eng_" in fname]
if __name__ == "__main__":
    train_images = np.random.choice(images,2000,replace=False)
    val_images = [im for im in images if im not in train_images]

    d="train"
    DatasetCatalog.register("roadsignsv1_" + d, lambda d=d: get_annotation_dict_from_mask(train_images))
    MetadataCatalog.get("roadsignsv1_" + d).set(thing_classes=["ARABIC","ENGLISH"])
    d="val"
    DatasetCatalog.register("roadsignsv1_" + d, lambda d=d: get_annotation_dict_from_mask(val_images))
    MetadataCatalog.get("roadsignsv1_" + d).set(thing_classes=["ARABIC","ENGLISH"])

    roadsign_metadata = MetadataCatalog.get("roadsignsv1_train")




    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("roadsignsv1_train",)
    cfg.DATASETS.TEST = ("roadsignsv1_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = "model_final_280758.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 15000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.TEST.EVAL_PERIOD = 100
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()