import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyDataset, MyDataLoader

class LoadData():
    def __init__(self, factor, label, batch_size=32, shuffle=False, num_workers=0, train_size=0.5, test_size=0.1):
        self.dataset = MyDataset(factor, label, train_size=train_size)
        self.dataloader = MyDataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, test_size=test_size)
        self.label_mean_val = self.dataset.label_mean_val
        self.label_std_val = self.dataset.label_std_val

    def get_train_data(self):
        return self.dataloader.get_train_loader()

    def get_test_data(self):
        return self.dataloader.get_test_loader()



class criterion(nn.Module):
    def __init__(self):
        super(criterion, self).__init__()

    # loss function
    # the smaller the better
    def mse_loss(self, outputs, targets):
        return self.mse_loss(outputs, targets)

    # 计算非去均值的 R^2
    # the larger the better
    def nondemeaned_R2(self, outputs, targets):
        ss_total = torch.sum((targets) ** 2)
        ss_residual = torch.sum((targets - outputs) ** 2)
        return 1 - (ss_residual / ss_total) if ss_total != 0 else 0


class optimizer():
    def __init__(self, model, lr=0.001):
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)












