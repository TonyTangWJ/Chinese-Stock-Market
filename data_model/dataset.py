import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

# 示例自定义数据集
class MyDataset(Dataset):
    def __init__(self, factor, label):
        self.factor = factor
        self.label = label
        self.factor['trade_date'] = pd.to_datetime(self.factor['trade_date'], format='%Y%m')
        self.label['trade_date'] = pd.to_datetime(self.label['trade_date'], format='%Y%m')
        self.label['trade_date'] = self.label['trade_date'] - pd.DateOffset(months=1)
        self.data = pd.merge(self.factor, self.label, on=['ts_code', 'trade_date'], how='left')
        self.data = self.data.sort_values(['trade_date','ts_code'], ascending=[True, True])
        self.data.dropna(inplace=True)
        self.data.drop(columns = ['ts_code', 'trade_date'], inplace=True)
        self.data = self.data.applymap(lambda x: float(x))
        if torch.cuda.is_available():
            self.data = torch.tensor(self.data.values, dtype=torch.float32, device='cuda:0')
        else:
            self.data = torch.tensor(self.data.values, dtype=torch.float32, device='cpu')
        return

    def __len__(self):
        return len(self.data)

    # label: highest_return == 1, lowest_return == -1, close_return == 0
    def __getitem__(self, idx):
        self.factor = self.data[idx, :-3]
        self.highest_return = self.data[idx, -3]
        self.lowest_return = self.data[idx, -2]
        self.close_return = self.data[idx, -1]
        return self.factor, self.highest_return, self.lowest_return, self.close_return


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=False, num_workers=0, train_size=0.5, test_size=0.1):
        if train_size + test_size >= 1.0:
            raise ValueError("train_size + test_size must be less than 1.0")
        self.train_size = train_size
        self.test_size = test_size
        # super(MyDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_loader(self):
        train_indice = int(len(self.dataset) * self.train_size)
        indices = list(range(len(self.dataset)))
        train_indices = indices[:train_indice]
        train_subset = Subset(self.dataset, train_indices)
        return DataLoader(train_subset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        

    def get_test_loader(self):
        train_indice = int(len(self.dataset) * self.train_size)
        test_indice = train_indice + int(len(self.dataset) * self.test_size)
        indices = list(range(len(self.dataset)))
        test_indices = indices[train_indice:test_indice]
        test_subset = Subset(self.dataset, test_indices)
        return DataLoader(test_subset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)