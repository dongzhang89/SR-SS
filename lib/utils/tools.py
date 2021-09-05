import math
import random
import cv2
import numpy as np

import torch

from utils.config import cfg
import pydensecrf.densecrf as dcrf

def random_scale_and_msc(image, lbl, fixed_scales, scales, aug=True):
    """
    Random scale for data augmentation and get three fixed_scales for fuse scores
    """
    if aug: factor = random.uniform(scales[0], scales[1])
    else: factor = 1

    # img
    h, w = image.shape[1:3]
    img = [cv2.resize(temp, (int(w * factor), int(h * factor))) for temp in image[:]]
    img_75 = [cv2.resize(temp, (int(w * factor * fixed_scales[1]), int(h * factor * fixed_scales[1]))) for temp in image[:]]
    img_50 = [cv2.resize(temp, (int(w * factor * fixed_scales[0]), int(h * factor * fixed_scales[0]))) for temp in image[:]]

    # change (B, H, W, C) to (B, C, H, W)
    img = torch.from_numpy(np.array(img).transpose(0, 3, 1, 2))
    img_75 = torch.from_numpy(np.array(img_75).transpose(0, 3, 1, 2))
    img_50 = torch.from_numpy(np.array(img_50).transpose(0, 3, 1, 2))

    return img, img_75, img_50


def msc_label(lbl, s1, s2, s3):
    lbl[lbl == 255] = 0

    label = [cv2.resize(temp, (s1[3], s1[2]), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    label = torch.from_numpy(np.array(label))

    label_75 = [cv2.resize(temp, (s2[3], s2[2]), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    label_75 = torch.from_numpy(np.array(label_75))

    label_50 = [cv2.resize(temp, (s3[3], s3[2]), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    label_50 = torch.from_numpy(np.array(label_50))

    return label, label_75, label_50

def msc_label_same(lbl, s1):
    lbl[lbl == 255] = 0

    label = [cv2.resize(temp, (s1[3], s1[2]), interpolation=cv2.INTER_NEAREST) for temp in lbl[:]]
    label = torch.from_numpy(np.array(label))

    return label


def adjust_learning_rate(optimizer, iter, lr):
    """
    Change learning rate in optimizer.
    Return learning rate of resnet part for tensorboardX
    """
    if not cfg.TRAIN.IF_POLY_POLICY and iter % cfg.TRAIN.LR_DECAY_ITERS or iter == 0: return lr
    for param_group in optimizer.param_groups:
        if cfg.TRAIN.IF_POLY_POLICY and iter != 0:
            s = math.pow((1 - iter / cfg.TRAIN.MAX_ITERS), cfg.TRAIN.POWER)
            t = round(param_group['lr'] / lr) # t = 1 or 10
            # assert (t == 1 or t == 10)
            if not(t == 1 or t == 10):
                print("\n\n Not 1 or 10 when iter = %d, t = %f \n\n" % (iter, param_group['lr'] / lr))
            param_group['lr'] = t * cfg.TRAIN.LEARNING_RATE * s
        elif iter % cfg.TRAIN.LR_DECAY_ITERS == 0 and iter != 0:
            param_group['lr'] = 0.1 * param_group['lr']

    if cfg.TRAIN.IF_POLY_POLICY:
        return cfg.TRAIN.LEARNING_RATE * s
    else:
        return 0.1 * lr


# use code in https://github.com/DrSleep/tensorflow-deeplab-resnet/
def dense_crf(probs, n_classes, img=None, n_iters=10,
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

    Returns:
      Refined predictions after MAP inference.
    """
    c, h, w = probs.shape

    probs = probs.copy(order='C')  # Need a contiguous array.

    d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
    U = -np.log(probs)  # Unary potential.
    U = U.reshape((n_classes, -1))  # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)

    if img is not None:
        assert (img.shape[0:2] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img.copy(order='C'))
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return preds
