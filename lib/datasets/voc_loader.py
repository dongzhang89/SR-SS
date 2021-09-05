import os.path as osp
import random
import cv2
import numpy as np
import math
import random

from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, txt, cfg, num_classes, mode):
        self.cfg = cfg
        self.data_path = cfg.VOC_DATA
        self.num_classes = num_classes
        self.data_list = self.read_image_label_path(txt)
        self.mode = mode

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        origin_img, origin_label = self.get_example(item)
        mean = np.array([104.00698793, 116.66876762, 122.67891434]).reshape(1, 1, 3)
        if self.mode == 'train':
            img = cv2.resize(origin_img - mean, tuple(self.cfg.TRAIN.INPUT_SIZE)).astype(float)
            label = cv2.resize(origin_label, tuple(self.cfg.TRAIN.INPUT_SIZE), interpolation=cv2.INTER_NEAREST).astype(float)
            # label = origin_label
            p = random.uniform(0, 1)
            img, label = self.flip(img, label, p)
            return img, label
        elif self.mode == 'val':
            return origin_img.astype(float) - mean, origin_label.astype(float)

    def read_image_label_path(self, txt):
        f = open(txt, "r")
        data_list = []
        while True:
            line = f.readline()
            if not line or line == '\n': break
            image_path, label_path = line.rstrip('\n').split(' ')
            data_list.append([image_path, label_path])
        f.close()
        return data_list

    def get_example(self, item):
        """Returns the i-th original example without any change.
         """
        image_path, label_path = self.data_list[item]
        img = cv2.imread(self.data_path + image_path).astype(float)
        label = cv2.imread(self.data_path + label_path)[:, :, 0]
        # label[label == 255] = 0

        return img, label

    def flip(self, I, G, p):
        if p > 0.5:
            return np.fliplr(I).copy(), np.fliplr(G).copy()
        else:
            return I, G

    # def process_data(self, img, label):
    #     """Returns the i-th example after processing
    #     """
    #
    #     ### Reduce images mean of coco
    #     mean = np.array([104.00698793,116.66876762,122.67891434]).reshape(1, 1, 3)
    #     img_temp = img - mean
    #
    #     ### data augumentation
    #     if self.cfg.TRAIN.IF_AUG:
    #
    #         factor = random.uniform(self.cfg.TRAIN.SCALES[0], self.cfg.TRAIN.SCALES[1])
    #         h, w = img_temp.shape[:2]
    #         img = cv2.resize(img_temp, (int(h * factor), int(w * factor))).astype(float)
    #         h, w = compute_outsize(img)
    #         label = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST).astype(float)
    #
    #     ### multi scale
    #     if self.cfg.TRAIN.IF_MSC:
    #         h, w = img.shape[:2]
    #         img_75 = cv2.resize(img_temp, (int(h * 0.75), int(w * 0.75))).astype(float)
    #         h, w = compute_outsize(img_75)
    #         label_75 = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST).astype(float)
    #
    #         h, w = img.shape[:2]
    #         img_50 = cv2.resize(img_temp, (int(h * 0.5), int(w * 0.5))).astype(float)
    #         h, w = compute_outsize(img_50)
    #         label_50 = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST).astype(float)
    #
    #         return img, label, img_75, label_75, img_50, label_50
    #     else:
    #         return img, label

