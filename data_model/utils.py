import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyDataset, MyDataLoader

class LoadData():
    def __init__(self, factor, label, batch_size=32, shuffle=False, num_workers=0, train_size=0.5, test_size=0.1):
        self.dataset = MyDataset(factor, label)
        self.dataloader = MyDataLoader(self.dataset, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle, 
                                       num_workers=num_workers, 
                                       train_size=train_size, 
                                       test_size=test_size)

    def get_train_loader(self, train_size=None):
        if train_size is not None:
            self.dataloader.train_size = train_size
        _ = self.dataloader.get_train_loader(train_size=self.dataloader.train_size)
        self.label_mean = self.dataloader.label_mean
        self.label_std = self.dataloader.label_std
        return _

    def get_test_loader(self, test_size=None):
        if test_size is not None:
            self.dataloader.test_size = test_size
        return self.dataloader.get_test_loader(test_size=self.dataloader.test_size)



