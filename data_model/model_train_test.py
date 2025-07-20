import torch
import torch.nn as nn
from dataset import MyDataset
from torch.utils.data import DataLoader, SequentialSampler


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
    
    def train(self, train_set, criterion, optimizer):
        total_loss = 0.0
        for inputs, targets in train_set:
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_set)





