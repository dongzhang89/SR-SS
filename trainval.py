import os.path as osp
import os, sys
import argparse
import pprint
import datetime
import numpy as np
import shutil

import torch
from torch.utils.data import DataLoader
from torch import nn

import _init_path
from utils.config import cfg, cfg_from_file, cfg_from_list
from utils.tools import random_scale_and_msc, msc_label, adjust_learning_rate, msc_label_same
from datasets.voc_loader import VOCDataset
from models.deeplab import DeepLab
from models.losses import loss_calc, get_class_loss, get_batch_label_vector, get_seg_loss
from test import eval

Lambda1=0.2
Lambda2=0.8
Lambda3=1.0

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a deeplab network')
    parser.add_argument('--dataset', dest='dataset', default='pascal_voc', type=str, help='training dataset')
    parser.add_argument('--net', dest='net', default='res101', type=str, help='vgg16, res101')
    parser.add_argument('--disp_interval', dest='disp_interval', default=2, type=int, help='number of iterations to display')
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval', default=1000, type=int, help='number of save model and evaluate it')
    parser.add_argument('--save_dir', dest='save_dir', default="models", type=str, help='directory to save models')
    parser.add_argument('--resume', dest='resume', default=False, type=bool, help='If resume training')
    parser.add_argument('--log_dir', dest='log_dir', default='logs', type=str, help='directory to save logs')
    parser.add_argument('--use_tfboard', dest='use_tfboard', default=True, type=bool, help='if use tensorboardX')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set config keys')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    print('Called with args: ')
    print(args)

    args.cfg_file = "cfgs/{}.yml".format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    # sys.exit(0)
    # torch.backends.cudnn.benchmark = True # Due to the different sizes of VOC images
    if torch.cuda.is_available() and not cfg.CUDA:
        print("Warning: You have a CUDA device, so you should run on it")

    if args.dataset == 'pascal_voc':
        num_classes = 21
        dataset = VOCDataset(osp.join(cfg.DATA_DIR, 'train.txt') ,cfg, num_classes, 'train')
        valset = VOCDataset(osp.join(cfg.DATA_DIR, 'val.txt'), cfg, num_classes, 'val')

    trainloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    valloader = DataLoader(dataset=valset, batch_size=1, shuffle=False)

    net = DeepLab(num_classes)
    net.create_architecture()

    # Load pre-trained model
    if cfg.TRAIN.PRETRAINED_MODEL:
        print("Loading pretrained weights from %s" % (cfg.TRAIN.PRETRAINED_MODEL))
        state_dict = torch.load(cfg.TRAIN.PRETRAINED_MODEL)
        net.load_state_dict(state_dict, strict=False)

    lr = cfg.TRAIN.LEARNING_RATE
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    momentum = cfg.TRAIN.MOMENTUM
    iter_size = cfg.TRAIN.ITER_SIZE

    ### Setting learning rate
    params = []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if 'conv2d_list' in key:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': 10 * lr, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': 10 * lr, 'weight_decay': weight_decay}]
            else:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

    optimizer = torch.optim.SGD(params, momentum=momentum)

    criterion = nn.CrossEntropyLoss()
    bceloss = nn.BCELoss()
    if cfg.CUDA: net.cuda()
    net.float()
    net.eval()

    data_iter = trainloader.__iter__()
    star_iter = 0
    ### If resume training
    if args.resume:
        load_model = sorted(os.listdir(args.save_dir))[-1]
        checkpoint = torch.load(osp.join(args.save_dir, load_model))
        lr = checkpoint['lr']
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.SGD(params, momentum=momentum)
        star_iter = checkpoint['iter']
        print("load model for %s !" % load_model)

    ### Use tensorboardX
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(args.log_dir)
        shutil.rmtree(args.log_dir)
        os.mkdir(args.log_dir)

    # for i in range(): _, _ = data_iter.next()
    ### Begin Training
    for iter in range(star_iter, cfg.TRAIN.MAX_ITERS+1):
        try:
            image, label = data_iter.next()
        except StopIteration:
            del data_iter
            data_iter = trainloader.__iter__()
            image, label = data_iter.next()

        img, img_75, img_50 = random_scale_and_msc(image.numpy(), label.numpy(), cfg.TRAIN.FIXED_SCALES, cfg.TRAIN.SCALES)

        if cfg.CUDA:
            img = img.cuda().float()
        else:
            img = img.float()

        out, pred = net(img)

        # label, label_75, label_50 = msc_label(label.numpy(), list(out[0].size()), list(out[1].size()), list(out[2].size()))
        # label_vector = get_batch_label_vector(label)
        #
        # if cfg.CUDA:
        #     label, label_75, label_50 = label.cuda().long(), label_75.cuda().long(), label_50.cuda().long()
        # else:
        #     label, label_75, label_50 = label.long(), label_75.long(), label_50.long()

        label = msc_label_same(label.numpy(), list(out[0].size()))
        label_vector = get_batch_label_vector(label)

        if cfg.CUDA:
            label = label.cuda().long()
        else:
            label = label.long()

        loss1_classification = get_class_loss(label_vector, pred, loss_fun=bceloss) 
        loss2_segmentation = get_seg_loss(out, label)

        loss3_SR_L = get_class_loss(pred[0], pred[1:], loss_fun=bceloss) 
        temp_label = torch.argmax(out[1], dim=1)
        temp_out = out[2:]
        temp_out.append(out[0])
        loss4_SR_F = get_seg_loss(temp_out, temp_label) 
        loss = Lambda1*loss1_classification + Lambda2*loss2_segmentation + Lambda3*loss3_SR_L + Lambda3*loss4_SR_F
        loss = loss / iter_size
        loss.backward()
        optimizer.step()
        lr = adjust_learning_rate(optimizer, iter, lr)
        optimizer.zero_grad()

        if iter % args.disp_interval == 0:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(("Time:%s  [iter %4d/%4d]  loss: %.4f,  lr: %.6e" % (now, iter, cfg.TRAIN.MAX_ITERS, loss.item() * iter_size, lr)))
            if args.use_tfboard:
                info = {
                    'loss': loss.item(),
                    'lr': lr
                }
                logger.add_scalars("loss_lr", info, iter)

        if iter % args.checkpoint_interval == 0 and iter > 100000:
            save_path = osp.join(args.save_dir, 'VOC12_'+str(iter)+'.pth')
            torch.save({'model': net.state_dict(),
                        'iter': iter,
                        'lr': lr
                        }, save_path)

            print("save model: {}".format(save_path))

    if args.use_tfboard:
        logger.close()