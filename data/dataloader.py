import os
import shutil
import torch

from .. import config
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def read_csv_labels(fname):
    """
    读取fname来给标签字典返回一个文件名
    example:
    input: '000bec180eb18c7604dcecc8fe0dba07', 'boston_bull'/n '001513dfcb2ffafc82cccf4d8bbaba97', 'dingo'
    output: {'000bec180eb18c7604dcecc8fe0dba07': 'boston_bull', '001513dfcb2ffafc82cccf4d8bbaba97': 'dingo'}
    """
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]

    return dict((name, label) for name, label in tokens)


def copyfile(filename, target_dir):
    """将文件复制到目标目录"""

    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def train_valid_split(data_dir, labels, valid_ratio):
    """将验证集从原始训练集中拆分出来"""

    import collections
    n = collections.Counter(labels.values()).most_common()[-1][1]
    import math
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label


# labels = read_csv_labels(os.path.join(data_dir, 'labels.csv'))
# train_data, valid_data = train_valid_split(data_dir, labels, config['valid_ratio'])


class MyDataloader:

    def __init__(self):
        self.batch_size = config['batch_size']
        self.image_size = 224
        self.data_dir = os.path.join('..', 'data', 'kaggle_data_tiny')

    def train_transform(self):
        preprocessing = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]
        )

        return preprocessing

    def val_transform(self):
        preprocessing = transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]
        )
        return preprocessing

    def train_dataloader(self):
        train_ds = torchvision.datasets.ImageFolder(
            os.path.join(self.data_dir, 'train_valid_test', 'train'),
            transform=self.train_transform()
        )

        loader: DataLoader = DataLoader(
            train_ds,
            self.batch_size,
            shuffle=True,
            drop_last=True
        )

        return loader

    def val_dataloader(self):
        valid_ds = torchvision.datasets.ImageFolder(
            os.path.join(self.data_dir, 'train_valid_test', 'valid'),
            transform=self.val_transform())

        loader: DataLoader = DataLoader(
            valid_ds,
            self.batch_size,
            shuffle=False,
            drop_last=False
        )

        return loader
