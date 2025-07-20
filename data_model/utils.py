import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyDataset
from torch.utils.data import DataLoader

class data_transform(): 
    def __init__(self, dataset):
        self.dataset = dataset

    def normalize(self):
        # 这里可以添加标准化或归一化的逻辑
        self.dataset.data = (self.dataset.data - self.dataset.data.mean()) / self.dataset.data.std()
        return self.dataset

    def standardize(self):
        # 这里可以添加标准化的逻辑
        self.dataset.data = (self.dataset.data - self.dataset.data.min()) / (self.dataset.data.max() - self.dataset.data.min())
        return self.dataset

    def robust_scale(self):
        # 这里可以添加鲁棒缩放的逻辑
        median = self.dataset.data.median()
        q75, q25 = self.dataset.data.quantile(0.75), self.dataset.data.quantile(0.25)
        iqr = q75 - q25
        self.dataset.data = (self.dataset.data - median) / iqr
        return self.dataset

















