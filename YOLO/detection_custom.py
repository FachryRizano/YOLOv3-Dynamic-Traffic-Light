import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import tensorflow as tf
from model.utils import detect_realtime, Load_Yolo_model, detect_video
from model.configs import *

# image_path   = "./IMAGES/plate_2.jpg"
video_path   = r"C:\Users\Asus\Desktop\traffic 7\Kapolri_ Arus Mudik Berlangsung Aman dan Lancar.mp4"

yolo = Load_Yolo_model()
# converter = tf.lite.TFLiteConverter.from_keras_model(yolo)
# yolo = converter.convert()

# detect_image(yolo, image_path, "./IMAGES/plate_1_detect.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
detect_video(yolo, video_path, './model_data/test.mp4', input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
# detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))