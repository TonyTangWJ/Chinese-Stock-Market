import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyDataset, MyDataLoader
import numpy as np

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



class RiskMetrics:
    def __init__(self, risk_free_rate=0.01, data_frequency='monthly'):
        self.risk_free_rate = risk_free_rate
        
        if data_frequency == 'daily':
            self.periods_per_year = 252
        elif data_frequency == 'weekly':
            self.periods_per_year = 52
        elif data_frequency == 'monthly':
            self.periods_per_year = 12
        elif data_frequency == 'quarterly':
            self.periods_per_year = 4
        else:
            raise ValueError("Supported frequencies: 'daily', 'weekly', 'monthly', 'quarterly'")
    
    def calculate_metrics(self, returns):
        """计算年化风险指标"""
        # 剔除return为0的值
        returns = np.array(returns)
        returns = returns[returns != 0]
        
        # 基础统计
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # 年化指标
        annualized_return = mean_return * self.periods_per_year
        annualized_volatility = std_return * np.sqrt(self.periods_per_year)
        
        # 夏普比率
        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cumulative_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = peak - cumulative_returns
        max_drawdown = np.max(drawdown)
        
        # 其他指标
        win_rate = np.sum(returns > 0) / len(returns)
        
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'data_frequency': f'{self.periods_per_year} periods/year'
        }


class ClippedLeakyReLU(nn.Module):
    def __init__(self, min_val=-0.2, max_val=0.2, negative_slope=0.3):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.negative_slope = negative_slope
        
    def forward(self, x):
        x = torch.where(x >= 0, x, self.negative_slope * x)
        x = torch.clamp(x, self.min_val, self.max_val)
        return x

class ScaledTanh(nn.Module):
    def __init__(self, scale=0.2):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return torch.tanh(x) * self.scale


