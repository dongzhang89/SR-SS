from yacs.config import CfgNode as CN

dataset_dir = '/home/datasets/'  # the path of the dataset

_C = CN()

_C.MODEL = CN()

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.PROB = 0.5
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Dataset root path
_C.DATASETS.ROOT = dataset_dir
