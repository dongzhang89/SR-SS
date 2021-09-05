import argparse
import os.path as osp
import pprint
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
from torch import nn

import _init_path
from datasets.voc_loader import VOCDataset
from utils.config import cfg, cfg_from_file
from utils.tools import random_scale_and_msc, dense_crf
from models.deeplab import DeepLab

# colour map
# use code in https://github.com/DrSleep/tensorflow-deeplab-resnet/
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a deeplab network')
    parser.add_argument('--img_path', dest='img',  default='test.jpg', type=str, help='test image')
    parser.add_argument('--net', dest='net', default='res101', type=str, help='vgg16, res101')
    parser.add_argument('--model', dest='model', default="models/7189.pth", type=str, help='pretrained model')
    parser.add_argument('--gpu', dest='gpu', default=False, type=bool, help='if use gpu when test single image')
    args = parser.parse_args()
    return args

def vis(pred, num_classes):
    h, w = pred.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            index = pred[i, j]
            assert index <= num_classes
            output[i, j] = label_colours[index]
    cv2.imwrite("result.jpg", output)
    cv2.imshow('Semantic Segmentation Result', output)
    cv2.waitKey(0)


def test_single_image(origin_img, net, args, num_classes):
    mean = np.array([104.00698793, 116.66876762, 122.67891434]).reshape(1, 1, 3)
    img = origin_img - mean
    img = img[np.newaxis, :, :, :]
    with torch.no_grad():
        # image, label = data_iter.next()
        img, img_75, img_50 = random_scale_and_msc(img, 0, cfg.TRAIN.FIXED_SCALES, cfg.TRAIN.SCALES, False)
        if args.gpu:
            img, img_75, img_50 = img.cuda().float(), img_75.cuda().float(), img_50.cuda().float()
        else:
            img, img_75, img_50 = img.float(), img_75.float(), img_50.float()

        out = net(img, img_75, img_50)[-1]
        interp = nn.UpsamplingBilinear2d(size=(img.size()[2], img.size()[3]))
        softmax = nn.Softmax2d()
        pred = softmax(interp(out)).cpu().numpy()[0]
        pred_crf = dense_crf(probs=pred, n_classes=num_classes, img=origin_img.astype('uint8'))
        pred_crf = np.argmax(pred_crf, axis=2)
        return pred_crf


if __name__ == "__main__":
    args = parse_args()

    num_classes = 21
    net = DeepLab(num_classes)
    net.create_architecture()
    checkpoint = torch.load(args.model)
    net.load_state_dict(checkpoint['model'])
    # net.load_state_dict(checkpoint) # caffe
    if args.gpu: net = net.cuda()
    net.float()
    net.eval()

    img = cv2.imread(args.img).astype(float)

    pred = test_single_image(img, net, args, num_classes)
    vis(pred, num_classes)