from torch.utils import data
from .datasets.voc import VocSegDataset
from .transforms import build_transforms


def build_dataset(cfg, transforms, is_train=True):
    datasets = VocSegDataset(cfg, is_train, transforms)
    return datasets


def make_data_loader(cfg, is_train=True):
    if is_train:
        shuffle = True
    else:
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, is_train)

    num_workers = 8
    if is_train==True:
        data_loader = data.DataLoader(
            datasets, batch_size=4, shuffle=shuffle, num_workers=num_workers, pin_memory=True
        )
    else:
        data_loader = data.DataLoader(
        datasets, batch_size=1, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    return data_loader
