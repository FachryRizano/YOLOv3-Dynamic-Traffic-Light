import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import tensorflow as tf
from model.utils import detect_realtime, Load_Yolo_model
from model.configs import *

yolo = Load_Yolo_model()
# converter = tf.lite.TFLiteConverter.from_keras_model(yolo)
# yolo = converter.convert()

detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
