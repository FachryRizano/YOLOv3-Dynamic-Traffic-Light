import os
import cv2
import random
import numpy as np
import tensorflow as tf
from model.utils import read_class_names, image_preprocess
from model.yolov3 import bbox_iou
from model.configs import *
from imgaug import augmenters as iaa 
from imgaug.augmentables.bbs import BoundingBoxesOnImage




class Dataset(object):
    # Dataset preprocess implementation
    def __init__(self, dataset_type, TEST_INPUT_SIZE=TEST_INPUT_SIZE):
        self.annot_path  = TRAIN_ANNOT_PATH if dataset_type == 'train' else TEST_ANNOT_PATH
        self.input_sizes = TRAIN_INPUT_SIZE if dataset_type == 'train' else TEST_INPUT_SIZE
        self.batch_size  = TRAIN_BATCH_SIZE if dataset_type == 'train' else TEST_BATCH_SIZE
        self.data_aug    = TRAIN_DATA_AUG   if dataset_type == 'train' else TEST_DATA_AUG

        self.train_input_sizes = TRAIN_INPUT_SIZE
        self.strides = np.array(YOLO_STRIDES)
        self.classes = read_class_names(TRAIN_CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = (np.array(YOLO_ANCHORS).T/self.strides).T
        self.anchor_per_scale = YOLO_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0


    def load_annotations(self, dataset_type):
        final_annotations = []
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        
        for annotation in annotations:
            # fully parse annotations
            line = annotation.split()
            image_path, index = "", 1
            for i, one_line in enumerate(line):
                if not one_line.replace(",","").isnumeric():
                    if image_path != "": image_path += " "
                    image_path += one_line
                else:
                    index = i
                    break
            if not os.path.exists(image_path):
                raise KeyError("%s does not exist ... " %image_path)
            if TRAIN_LOAD_IMAGES_TO_RAM:
                image = cv2.imread(image_path)
            else:
                image = ''
            final_annotations.append([image_path, line[index:], image])
        return final_annotations

    def __iter__(self):
        return self

    def Delete_bad_annotation(self, bad_annotation):
        print(f'Deleting {bad_annotation} annotation line')
        bad_image_path = bad_annotation[0]
        bad_image_name = bad_annotation[0].split('/')[-1] # can be used to delete bad image
        bad_xml_path = bad_annotation[0][:-3]+'xml' # can be used to delete bad xml file

        # remove bad annotation line from annotation file
        with open(self.annot_path, "r+") as f:
            d = f.readlines()
            f.seek(0)
            for i in d:
                if bad_image_name not in i:
                    f.write(i)
            f.truncate()
    
    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice([self.train_input_sizes])
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            exceptions = False
            num = 0
            #annotation more than 1
            #random <0.5

            #ambil gambar pertama
            #ambil gambar kedua
            #implement fungsi mix up
            # 
            def mixup(image1, bboxes1, image2, bboxes2, ratio=MIX_UP_THRESHOLD):
                '''
                    Mixup 2 image
                    
                    image_info_1, image_info_2: Info dict 2 image with keys = {"image", "label", "box", "difficult"}
                    lambd: Mixup ratio
                    
                    Out: mix_image (Temsor), mix_boxes, mix_labels, mix_difficulties
                '''
                mixup_width = max(image1.shape[1], image2.shape[1])
                mix_up_height = max(image1.shape[0], image2.shape[0])
                
                mix_img = np.zeros((mix_up_height, mixup_width,3),dtype=float)
                mix_img[:, :image1.shape[0], :image1.shape[1]] = image1 * ratio
                mix_img[:, :image2.shape[0], :image2.shape[1]] += image2 * (1. - ratio)
            
                mix_boxes = np.concatenate((bboxes1,bboxes2), axis= 0)
                
                return mix_img, mix_boxes
                
            # curr_dict = {"image" : annotation[2], "box" : annotation[2]}
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    if MIX_UP:
                        if num > 0:
                            #apply mixup function here with 50% chance
                            if random.random()<0.5:
                                #annotation = [image_path,bboxes,image]
                                prev_annotation = self.annotations[index-1]
                                prev_image, prev_bboxes = self.parse_annotation(prev_annotation, only_parse=True)
                                image, bboxes = mixup(image, bboxes, prev_image, prev_bboxes)
                                # image = cv2.cvtColor(image.astype('float32'),cv2.COLOR_BGR2RGB)
                                
                    try:
                        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                    except IndexError:
                        exceptions = True
                        self.Delete_bad_annotation(annotation)
                        print("IndexError, something wrong with", annotation[0], "removed this line from annotation file")

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1

                if exceptions: 
                    print('\n')
                    raise Exception("There were problems with dataset, I fixed them, now restart the training process.")
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration


    # https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720562.pdf

    ##Augmentor with imgaug
    aug = iaa.SomeOf(2, [
    #top 3 augment technique from paper
    iaa.pillike.Equalize(),
    iaa.Affine(translate_percent={"y":(-1, 1)}),
    iaa.Rotate(),

    # iaa.Affine(scale=(0.5, 1.5)),
    iaa.Sharpen(alpha=(0,1.0)),
    iaa.Posterize(),
    iaa.Solarize(0.5),
    iaa.pillike.Autocontrast(0.5),
    iaa.Affine(translate_percent={"x":(-1, 1),"y":(-1, 1)}),
    iaa.Affine(translate_percent={"x":(-1, 1)}),
    iaa.imgcorruptlike.Contrast(),
    iaa.imgcorruptlike.Brightness(),
    iaa.ShearY()

    ])

    def aug_with_imgaug(self,image, bboxes,aug = aug):
        if random.random() < 0.5:
            bbs = BoundingBoxesOnImage.from_xyxy_array(bboxes[:,:-1], shape= image.shape)
            image, bbs = aug(image=image, bounding_boxes=bbs)
            #disregard bounding boxes which have fallen out of image pane    
            bbs = bbs.remove_out_of_image()

            #clip bounding boxes which are partially outside of image pane
            bbs = bbs.clip_out_of_image()
            bboxes = np.column_stack((bbs.to_xyxy_array(),bboxes[:,-1][:bbs.to_xyxy_array().shape[0],np.newaxis])).astype(int)
            
        return image, bboxes
    
    def parse_annotation(self, annotation, mAP = 'False', only_parse= False):
        
        if TRAIN_LOAD_IMAGES_TO_RAM:
            image_path = annotation[0]
            image = annotation[2]
        else:
            image_path = annotation[0]
            image = cv2.imread(image_path)
            
        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])
        if not only_parse:
          if self.data_aug:
              image, bboxes = self.aug_with_imgaug(np.copy(image), np.copy(bboxes))
              # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mAP == True: 
            return image, bboxes
        
        image, bboxes = image_preprocess(np.copy(image), [self.input_sizes, self.input_sizes], np.copy(bboxes))
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
