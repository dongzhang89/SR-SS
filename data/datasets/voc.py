import os
import numpy as np
from PIL import Image
from torch.utils import data


def read_images(root, train):
    txt_fname = os.path.join(root) + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'images', i + '.jpg') for i in images]
    label = [os.path.join(root, 'segmentations', i + '.png') for i in images]
    return data, label

def rfuse(image, label):
    image_onehot = np.array(label)
    image_onehot = np.where(image_onehot > 0, 1, 0)
    r, g, b = image.split()
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    r = np.multiply(r, image_onehot)
    g = np.multiply(g, image_onehot)
    b = np.multiply(b, image_onehot)
    im_r = Image.fromarray(np.uint8(r))
    im_g = Image.fromarray(np.uint8(g))
    im_b = Image.fromarray(np.uint8(b))
    im = Image.merge('RGB', (im_r, im_g, im_b))
    return im

class VocSegDataset(data.Dataset):

    def __init__(self, cfg, train, transforms=None):
        self.cfg = cfg
        self.train = train
        self.transforms = transforms
        self.data_list, self.label_list = read_images(self.cfg.DATASETS.ROOT, train) #,self.mask,self.position

    def __getitem__(self, item):
        img = self.data_list[item]
        label = self.label_list[item]
        img = Image.open(img)
        label = Image.open(label)
        img, label = self.transforms(img, label)
        return img, label

    def __len__(self):
        return len(self.data_list)