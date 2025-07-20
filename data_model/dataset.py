import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

class MyDataset(Dataset):
    def __init__(self, factor, label, train_size=0.5):
        self.factor = factor
        self.label = label
        self.train_size = train_size
        # merge factor and label on 'ts_code' and 'trade_date'
        self.factor['trade_date'] = pd.to_datetime(self.factor['trade_date'], format='%Y%m')
        self.label['trade_date'] = pd.to_datetime(self.label['trade_date'], format='%Y%m')
        self.label['trade_date'] = self.label['trade_date'] - pd.DateOffset(months=1)
        self.data = pd.merge(self.factor, self.label, on=['ts_code', 'trade_date'], how='left')
        self.data = self.data.sort_values(['trade_date','ts_code'], ascending=[True, True])
        self.data.dropna(inplace=True)
        self.data.drop(columns = ['ts_code', 'trade_date'], inplace=True)
        self.data = self.data.applymap(lambda x: float(x))
        # winsorize the data
        self.data = self.winsorize_data(self.data, lower=0.04, upper=0.96)
        # convert to tensor
        if torch.cuda.is_available():
            self.data = torch.tensor(self.data.values, dtype=torch.float32, device='cuda:0')
        else:
            self.data = torch.tensor(self.data.values, dtype=torch.float32, device='cpu')
        # get normalization parameters
        self.get_factors_norm_params(train_size=self.train_size)
        self.get_labels_norm_params(train_size=self.train_size)
        return

    def __len__(self):
        return len(self.data)

    # factor, highest_return, lowest_return, close_return
    def __getitem__(self, idx):
        self.factor = self.data[idx, :-3]
        self.highest_return = self.data[idx, -3]
        self.lowest_return = self.data[idx, -2]
        self.close_return = self.data[idx, -1]
        return self.factor, self.highest_return, self.lowest_return, self.close_return
    
    def get_factors_norm_params(self, train_size):
        train_indice = int(len(self.data) * train_size)
        train_data = self.data[:train_indice, :-3]  # 取特征部分（除最后3列）
        self.factor_min = train_data.min(dim=0, keepdim=True)[0]  # 保持维度 [1, n_features]
        self.factor_max = train_data.max(dim=0, keepdim=True)[0]  # [1, n_features]
        self.factor_mean = train_data.mean(dim=0, keepdim=True)   # [1, n_features]
        self.factor_std = train_data.std(dim=0, keepdim=True)     # [1, n_features]
        return
        
    def get_labels_norm_params(self, train_size):
        train_indice = int(len(self.data) * train_size)
        train_data = self.data[:train_indice, -3:]  # 取最后3列作为label
        self.label_min = train_data.min(dim=0, keepdim=True)[0]  # [1, 3]
        self.label_max = train_data.max(dim=0, keepdim=True)[0]  # [1, 3]
        self.label_mean = train_data.mean(dim=0, keepdim=True)   # [1, 3]
        self.label_std = train_data.std(dim=0, keepdim=True)     # [1, 3]
        return
    

    def winsorize_data(self, df, lower=0.04, upper=0.96):
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            lower_bound = df[col].quantile(lower)  # 4% 分位数
            upper_bound = df[col].quantile(upper)  # 96% 分位数
            df[col] = df[col].clip(lower_bound, upper_bound)  # 限制在 [4%, 96%]
        return df



class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=False, num_workers=0, test_size=0.1):
        self.dataset = dataset
        if dataset.train_size + test_size > 1.0:
            raise ValueError("train_size + test_size must be less than 1.0")
        self.train_size = self.dataset.train_size
        self.test_size = test_size
        # super(MyDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.factor_min = self.dataset.factor_min
        self.factor_max = self.dataset.factor_max
        self.factor_mean = self.dataset.factor_mean
        self.factor_std = self.dataset.factor_std
        self.label_min = self.dataset.label_min
        self.label_max = self.dataset.label_max
        self.label_mean = self.dataset.label_mean
        self.label_std = self.dataset.label_std
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_loader(self):
        train_indice = int(len(self.dataset) * self.train_size)
        indices = list(range(len(self.dataset)))
        train_indices = indices[:train_indice]
        train_subset = Subset(self.dataset, train_indices)
        # normalize the factors
        train_subset_factors_norm = train_subset.dataset.data[:train_indice, :-3]
        train_subset_factors_norm = (train_subset_factors_norm - self.factor_mean) / (self.factor_std + 1e-8)
        train_subset.dataset.data[:train_indice, :-3] = train_subset_factors_norm
        # normalize the targets
        train_subset_labels_norm = train_subset.dataset.data[:train_indice, -3:]
        train_subset_labels_norm = (train_subset_labels_norm - self.label_mean) / (self.label_std + 1e-8)
        train_subset.dataset.data[:train_indice, -3:] = train_subset_labels_norm
        return DataLoader(train_subset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        

    def get_test_loader(self):
        train_indice = int(len(self.dataset) * self.train_size)
        test_indice = train_indice + int(len(self.dataset) * self.test_size)
        indices = list(range(len(self.dataset)))
        test_indices = indices[train_indice:test_indice]
        test_subset = Subset(self.dataset, test_indices)
        # normalize the factors
        test_subset_factors_norm = test_subset.dataset.data[train_indice:test_indice, :-3]
        test_subset_factors_norm = (test_subset_factors_norm - self.factor_mean) / (self.factor_std + 1e-8)
        test_subset.dataset.data[train_indice:test_indice, :-3] = test_subset_factors_norm
        # normalize the targets
        test_subset_labels_norm = test_subset.dataset.data[train_indice:test_indice, -3:]
        test_subset_labels_norm = (test_subset_labels_norm - self.label_mean) / (self.label_std + 1e-8)
        test_subset.dataset.data[train_indice:test_indice, -3:] = test_subset_labels_norm
        return DataLoader(test_subset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

