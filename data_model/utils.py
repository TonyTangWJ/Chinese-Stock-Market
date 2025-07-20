import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyDataset, MyDataLoader

class LoadData():
    def __init__(self, factor, label, batch_size=32, shuffle=False, num_workers=0, train_size=0.5, test_size=0.1):
        self.dataset = MyDataset(factor, label, train_size=train_size)
        self.dataloader = MyDataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, test_size=test_size)
        self.label_mean = self.dataset.label_mean
        self.label_std = self.dataset.label_std

    def get_train_loader(self):
        return self.dataloader.get_train_loader()

    def get_test_loader(self):
        return self.dataloader.get_test_loader()



