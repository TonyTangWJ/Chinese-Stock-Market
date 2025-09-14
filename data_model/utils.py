import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import MyDataset, MyDataLoader
import numpy as np
import math

class LoadData():
    def __init__(self, factor, label, batch_size=32, shuffle=True, num_workers=0, train_size=0.5, test_size=0.1):
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
        cumulative_returns = np.cumprod(1 + returns/100) - 1 
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1e-8)
        max_drawdown = -np.max(drawdown)
        
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


class MAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        
        # 确保d_model能被nhead整除
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        
        # Q, K, V线性投影层
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        self.dropout_layer = nn.Dropout(dropout) if dropout is not None else nn.Identity()


    def forward(self, x):
        """
        二维Feature-wise Attention
        
        参数:
            x: 输入张量 [batch_size, d_model]
            
        返回:
            output: 输出张量 [batch_size, d_model]
        """
        batch_size, d_model = x.shape
        
        # 1. 生成Q, K, V
        Q = self.w_q(x)  # [batch_size, d_model]
        K = self.w_k(x)  # [batch_size, d_model]
        V = self.w_v(x)  # [batch_size, d_model]

        if self.nhead > 1:
            # 2. 重塑为多头形式
            # Q: [batch_size, nhead, d_k]
            # K: [batch_size, nhead, d_k]
            Q = Q.view(batch_size, self.nhead, self.d_k)
            K = K.view(batch_size, self.nhead, self.d_k)
            V = V.view(batch_size, self.nhead, self.d_k)
        
            # 3. 计算注意力分数 - 在特征维度上计算
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # 4. 计算注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, nhead]
            attn_weights = self.dropout_layer(attn_weights) if self.dropout_layer is not None else attn_weights

            # 5. 加权求和
            output = torch.matmul(attn_weights, V) 

            # 6. 拼接多头输出
            output = output.transpose(1, 2).contiguous().view(batch_size, -1)  # [batch_size, d_model]
            
            # 7. 输出投影
            output = self.w_o(output)
        else:
            # 3. 计算注意力分数 - 在特征维度上计算
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # 4. 计算注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, nhead]
            attn_weights = self.dropout_layer(attn_weights) if self.dropout_layer is not None else attn_weights

            # 5. 加权求和
            output = torch.matmul(attn_weights, V) 
            
            # 7. 输出投影
            output = self.w_o(output)

        return output


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, dropout=None):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model*2)
        self.w2 = nn.Linear(d_model*2, d_model)
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        
    def forward(self, x):
        x = self.w1(x)
        x = nn.LeakyReLU(0.3)(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=None):
        super().__init__()
        self.self_attn = MAttention(d_model, nhead, dropout=dropout)
        self.ffn = PositionWiseFFN(d_model, dropout=dropout)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        
    def forward(self, x):
        # 自注意力子层
        attn_output = self.self_attn(x)
        x = x + attn_output
        x = self.norm1(x)
        
        # 前馈网络子层
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=None):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
