import cv2
import open3d as o3d
import numpy as np
import torch

from RealsenYolo import RealsenseYOLO
from RGBDFusion import RGBDFusion

class Config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo_model = 'yolov8n.pt'
    fastsam_model = 'fastsam-s.pt'
    detect_classes = ['cup']
    bag_file = "20251112_135756.bag"

config = Config()

try:
    realsense_yolo = RealsenseYOLO(
        model_path = config.yolo_model,
        bag_path = config.bag_file,
        detect_classes=config.detect_classes,
        device=config.device,
    )
    print("Khởi tạo Yolov8 thành công với luồng RBG")

    rgbdfusion = RGBDFusion(
        fastsam_ckpt= config.fastsam_model,
        device = config.device,
    )
except Exception as e: