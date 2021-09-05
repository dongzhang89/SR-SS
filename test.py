import argparse
import os.path as osp
import pprint
import numpy as np

import _init_path
import torch
from torch.utils.data import DataLoader
from torch import nn

from datasets.voc_loader import VOCDataset
from utils.config import cfg, cfg_from_file
from utils.tools import random_scale_and_msc, msc_label, dense_crf
from models.deeplab import DeepLab
from collections import OrderedDict
import tqdm



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a deeplab network')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
    parser.add_argument('--model', dest='model',
                      help='pretrained model', default="models/7310.pth",
                      type=str)
    args = parser.parse_args()
    return args

def compute_iou(label, pred, num_classes):
    label = label.flatten()
    pred = pred.flatten()

    k = (label >= 0) & (label < num_classes)
    return np.bincount((num_classes * label[k] + pred[k]).astype(int), minlength=num_classes**2).reshape(num_classes, num_classes)

def eval(datalodaer, net, hist):
    with torch.no_grad():
        for i, (image, label) in enumerate(datalodaer):
            # image, label = data_iter.next()
            image = image.numpy()[0]
            squares = np.zeros((513, 513, 3))
            squares[:image.shape[0], :image.shape[1], :] = image
            squares = squares[np.newaxis, :, :, :]
            img, img_75, img_50 = random_scale_and_msc(squares, label.numpy(), cfg.TRAIN.FIXED_SCALES, cfg.TRAIN.SCALES, False)
            if cfg.CUDA:
                img, img_75, img_50 = img.cuda().float(), img_75.cuda().float(), img_50.cuda().float()
            else:
                img, img_75, img_50 = img.float(), img_75.float(), img_50.float()

            out = net(img, img_75, img_50)[-1]
            mean = np.array([104.00698793, 116.66876762, 122.67891434]).reshape(1, 1, 3)
            interp = nn.UpsamplingBilinear2d(size=(img.size()[2], img.size()[3]))
            if cfg.TEST.IF_CRF:
                softmax = nn.Softmax2d()
                pred = softmax(interp(out)).cpu().numpy()[0]
                pred = pred[:, :image.shape[0], :image.shape[1]]
                origin_img = image + mean
                pred_crf = dense_crf(probs=pred, n_classes=num_classes, img=origin_img.astype('uint8'))
                pred_crf = np.argmax(pred_crf, axis=2)

                hist += compute_iou(label.numpy()[0, :, :], pred_crf, num_classes)
            else:
                pred = interp(out).cpu().numpy()[0]
                pred = pred[:, :image.shape[0], :image.shape[1]]
                pred = np.argmax(pred.transpose(1, 2, 0), axis=2)

                hist += compute_iou(label.numpy()[0, :, :], pred, num_classes)
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("Mean iou = %.2f%%" % (np.sum(miou) * 100 / len(miou)))


if __name__ == "__main__":
    args = parse_args()
    print('Called with args: ')
    print(args)

    args.cfg_file = "cfgs/{}.yml".format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if args.dataset == 'pascal_voc':
        num_classes = 21
        valset = VOCDataset(osp.join(cfg.DATA_DIR, 'val.txt'), cfg, num_classes, 'val')

    valloader = DataLoader(dataset=valset, batch_size=1, shuffle=False)
    # data_iter = iter(valloader)
    net = DeepLab(num_classes)
    net.create_architecture()
    if cfg.CUDA: net = net.cuda()
    net.eval()

    # for i in range(30, 41):
    #     # model = args.model
    #     model = 'dmodels/VOC12_%d000.pth' % int(i)
    #     checkpoint = torch.load(model)
    #     # net.load_state_dict(checkpoint)
    #     net.load_state_dict(checkpoint['model'])
    #     hist = np.zeros((num_classes, num_classes))
    #     print("Model Path%s: " % model)
    #     eval(valloader, net, hist)

    model = args.model
    # model = 'dmodels/VOC12_%d000.pth' % int(i)
    checkpoint = torch.load(model)
    # net.load_state_dict(checkpoint)
    net.load_state_dict(checkpoint['model'])
    hist = np.zeros((num_classes, num_classes))
    print("Model Path: %s " % model)
    eval(valloader, net, hist)