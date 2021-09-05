from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# Data directory
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# __C.VOC_DATA = osp.abspath(osp.join(__C.DATA_DIR, 'VOC2012'))
__C.VOC_DATA = "/Users/a58/workspace/dataset/voc_dataset/VOCdevkit/VOC2012"

#
# Training options
#
__C.TRAIN = edict()

# Pretrained model
__C.TRAIN.PRETRAINED_MODEL = osp.abspath(osp.join(__C.DATA_DIR, 'DeepLab_resnet_pretrained.pth'))
# __C.TRAIN.PRETRAINED_MODEL = ''

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001
# Momentum
__C.TRAIN.MOMENTUM = 0.9
# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005
# Max iters
__C.TRAIN.MAX_ITERS = 40000
# Mini batch size for training
__C.TRAIN.BATCH_SIZE = 2
# Num iters to accumulate gradients
__C.TRAIN.ITER_SIZE = 10
# If use learning rate policy
__C.TRAIN.IF_POLY_POLICY = True
# if true
__C.TRAIN.POWER = 0.9
# if false
__C.TRAIN.LR_DECAY_ITERS = 2000
# If rescale images to different sizes
__C.TRAIN.IF_MSC = True
# Multi scale
__C.TRAIN.FIXED_SCALES = [0.5, 0.75, 1]
# Data augmentation
__C.TRAIN.IF_AUG = True
__C.TRAIN.SCALES = [0.5, 1.5]
# Data input size
__C.TRAIN.INPUT_SIZE = [321, 321]
# Whether to initialize the weights with truncated normal distribution
__C.TRAIN.TRUNCATED = False



# Data


#
# Testing options
#
__C.TEST = edict()

# If use CRF
__C.TEST.IF_CRF = False
__C.TEST.SIGMA = 3
__C.TEST.OMEGA = 3


# If use GPU for training
__C.CUDA = False



def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value