import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


class MyDataloader:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.image_size = 224
        self.data_dir = os.path.join('dataset', 'dog_breed_identification')

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
