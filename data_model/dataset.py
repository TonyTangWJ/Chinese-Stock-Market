import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

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
            self.data = torch.tensor(self.data.values, dtype=torch.float32).cuda()
        else:
            self.data = torch.tensor(self.data.values, dtype=torch.float32)
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

