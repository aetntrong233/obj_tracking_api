import os
cwd = os.path.abspath(os.getcwd())
import sys
sys.path.append(os.path.join(cwd, 'detector'))
import cv2
import numpy as np
import random


DEFAULT_WEIGHTS = os.path.normpath(os.path.join(cwd, 'detector/weights/model.weights'))
DEFAULT_CFG = os.path.normpath(os.path.join(cwd, 'detector/weights/configs.cfg'))
DEFAULT_NAMES_PATH = os.path.normpath(os.path.join(cwd, 'detector/weights/coco.names'))

class Detector(object):
    def __init__(self, is_cuda=False, weights=DEFAULT_WEIGHTS, cfg=DEFAULT_CFG, img_size=416, names_path=DEFAULT_NAMES_PATH):
        self.is_cuda = is_cuda
        self.weights = weights
        self.cfg = cfg
        self.img_size = img_size
        self.model = self.load_model()
        self.names = self.load_classes(names_path)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def load_classes(self, path):
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))

    def load_model(self):
        net = cv2.dnn.readNet(self.weights, self.cfg)
        if self.is_cuda:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)   
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1./255, swapRB=True)
        return model

    def inference(self, src_img, conf_thres=0.25, iou_thres=0.45, *args, **kwargs):
        h, w = src_img.shape[:2]
        classIds, confidences, boxes = self.model.detect(src_img.copy(), conf_thres, iou_thres)
        ret = []
        for (classid, confidence, box) in zip(classIds, confidences, boxes):
            temp = {}
            temp['bbox'] = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
            temp['box'] = [box[0]/w, box[1]/h, (box[0]+box[2])/w, (box[1]+box[3])/h]
            temp['confidence'] = float(confidence)
            temp['class_num'] = int(classid)
            temp['class'] = self.names[temp['class_num']]
            ret.append(temp)
        return ret

    def draw(self, src_img, classes=None, line_thickness=3):
        img = src_img.copy()
        pred = self.inference(src_img, classes=classes)
        for det in pred:
            cv2.rectangle(img, (int(det['bbox'][0]),int(det['bbox'][1])), (int(det['bbox'][2]),int(det['bbox'][3])), self.colors[det['class_num']], line_thickness)
            cv2.putText(img , det['class'], (int(det['bbox'][2]+10),int(det['bbox'][3])), 0, 0.5, self.colors[det['class_num']])
        return img