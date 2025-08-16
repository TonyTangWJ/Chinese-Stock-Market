import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
import copy
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim=1, target=3, seq_len=60, d_model=62, 
                 nhead=1, num_layers=3, d_ff=256, dropout=0, p=10000,
                 alpha=0.8, l1_ratio=0.5, model_name="Transformer"):
        """
        Transformer模型用于金融时序预测
        
        参数:
            input_dim: 输入特征维度
            output_dim: 输出维度 (默认1)
            target: 目标列索引 (默认3)
            seq_len: 序列长度 (默认60)
            d_model: 模型维度 (默认62)
            nhead: 注意力头数 (默认1)
            num_layers: Transformer层数 (默认3)
            d_ff: 前馈网络维度 (默认256)
            dropout: Dropout比例 (默认0)
            alpha: 弹性网络正则化强度 (默认0.8)
            l1_ratio: L1正则化比例 (默认0.5)
            model_name: 模型名称 (默认"Transformer")
        """
        super(Transformer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target = target
        self.seq_len = seq_len
        self.d_model = d_model
        self.p = p
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.model_name = f"{model_name}_{num_layers}Layers_{d_ff}FF_{p}P_{nhead}Heads"
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        # 标准化
        self.input_norm = nn.LayerNorm(self.input_dim)
        # 输入映射层
        self.input_proj = nn.Linear(self.input_dim, self.d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=seq_len, p=self.p)
        
        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(
            self.d_model, self.nhead, self.num_layers, self.d_ff, dropout=self.dropout
        )
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_ff // 2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff // 2, self.output_dim)
        )
        
        # 初始化
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        
        # 设备设置
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)
        
        # 模型保存路径
        if not os.path.exists("model/checkpoints"):
            os.makedirs("model/checkpoints")
        if not os.path.exists("model/final_models"):
            os.makedirs("model/final_models")
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pth"
        self.model_path = f"model/final_models/{self.model_name}.pth"

    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, seq_len, input_dim)
            解释:       (股票数量, 时间步长, 特征维度)
        """
        batch_size = x.size(0)

        # 确保序列长度正确
        if x.size(1) != self.seq_len:
            if x.size(1) < self.seq_len:
                # 如果序列太短，用0填充
                padding_length = self.seq_len - x.size(1)
                padding = torch.zeros(batch_size, padding_length, x.size(2), device=x.device)
                x = torch.cat([x, padding], dim=1)  # 在序列末尾填充0
            else:
                # 如果序列太长，截取最后seq_len个时间步
                x = x[:, -self.seq_len:, :]
        
        # 输入投影: (batch_size, seq_len, input_dim) -> (batch_size, seq_len, d_model)
        x = self.input_norm(x)
        x = self.input_proj(x, self.d_model)
        '''
        # encoding到0-1变量
        x = self.input_activation(x)
        '''
        # 添加位置编码
        x = self.positional_encoding(x)
        # Transformer编码器
        x = self.transformer_encoder(x)
        # 输出投影
        x = self.output_proj(x)
        return x

    def elastic_net_loss(self, outputs, targets):
        """弹性网络损失函数：MSE + L1正则化 + L2正则化"""
        mse_loss = self.loss_fn(outputs, targets)
        l1_reg = 0
        l2_reg = 0
        
        # 对所有线性层应用正则化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                l1_reg += torch.sum(torch.abs(module.weight))
                l2_reg += torch.sum(module.weight ** 2)
        
        # 弹性网络正则化项
        elastic_reg = self.alpha * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg)
        
        return mse_loss + elastic_reg

    def train_step(self, train_loader, criterion=None):
        """训练步骤"""
        if criterion is None:
            criterion = self.loss_fn
        elif criterion == 'elastic_net':
            criterion = self.elastic_net_loss
        else:
            pass
        
        self.train()
        total_loss = 0.0
        
        for train_data in train_loader:
            inputs = train_data[0].to(self.device)
            targets = train_data[self.target].to(self.device)
            
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def fit(self, train_loader, epochs=100, resume_training=False, patience=5, criterion=None):
        """训练主函数"""
        start_epoch = 0
        no_improvement_count = 0
        
        if resume_training:
            start_epoch = self._load_checkpoint()
            print(f"Resuming training from epoch {start_epoch + 1}")

        prev_loss = float('inf')
        for epoch in tqdm(range(start_epoch, epochs), colour='#FA6780'):
            train_loss = self.train_step(train_loader, criterion)
            
            # 每10个epoch保存checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")

            # 早停机制
            if round(train_loss, 4) < round(prev_loss, 4) - 0.001:
                prev_loss = train_loss
                no_improvement_count = 0
                self.save_model()
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    break

    def predict(self, test_loader, label_mean=None, label_std=None):
        
        self.eval()
        pred_list = []
        act_list = []
        
        with torch.no_grad():
            for test_data in test_loader:
                data = test_data[0].to(self.device)      # [batch_size, timesteps, features]
                act_high = test_data[1].to(self.device)  # [batch_size, timesteps]
                act_low = test_data[2].to(self.device)   # [batch_size, timesteps]
                act_close = test_data[3].to(self.device) # [batch_size, timesteps]
                
                batch_size, timesteps, _ = data.shape
                    
                # 获取当前时间步的真实标签
                current_act_high = act_high[:, 1].unsqueeze(1)
                current_act_low = act_low[:, 2].unsqueeze(1)
                current_act_close = act_close[:, 3].unsqueeze(1)
                
                # 组合实际值
                current_act = torch.cat([current_act_high, current_act_low, current_act_close], dim=1)
                
                # 反标准化
                if label_mean is not None and label_std is not None:
                    current_act = current_act * label_std + label_mean
                
                act_list.append(current_act.cpu().numpy())
                
                # 预测
                pred = self(data)  # [batch_size, 1]
                
                # 反标准化预测值
                if label_mean is not None and label_std is not None:
                    pred = pred * label_std[:, self.target-1:self.target] + label_mean[:, self.target-1:self.target]
                
                pred_list.append(pred.cpu().numpy())
        
        pred = np.concatenate(pred_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        return pred, act

    def evaluate(self, test_loader):
        """nondemeaned R² evaluation"""
        self.eval()
        total_sse = 0.0
        total_ss = 0.0
        
        with torch.no_grad():
            for test_data in test_loader:
                inputs = test_data[0].to(self.device)
                targets = test_data[self.target].to(self.device)
                
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                
                outputs = self(inputs)
                
                # 确保非负
                targets = torch.clamp(targets, min=0)
                outputs = torch.clamp(outputs, min=0)
                
                # 计算误差
                sse = torch.sum((targets - outputs) ** 2)
                ss = torch.sum(targets ** 2)
                total_sse += sse.item()
                total_ss += ss.item()
        
        if total_ss < 1e-8:
            return 0
        else:
            return 1 - total_sse / total_ss

    def reset_model(self):
        """重置模型"""
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, path=None):
        """加载模型"""
        if path is None:
            path = self.model_path
        self.load_state_dict(torch.load(path, map_location=self.device))

    def _save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _load_checkpoint(self):
        """加载检查点"""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch']
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=500, p=10000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(p) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: Tensor, shape [batch_size, seq_len, d_model]
        返回:
            Tuple[Tensor, Tensor]: 
                - 添加位置编码后的张量(同输入形状)
                - 自动生成的掩码 [batch_size, seq_len]
        """
        # 自动生成掩码(任何特征维度上非零的位置视为有效)
        mask = (x.abs().sum(dim=-1) > 0).float()  # [batch_size, seq_len]
        # 获取位置编码(自动截取到输入序列长度)
        pe = self.pe[:, :x.size(1), :]  # [1, seq_len, d_model]
        # 应用掩码
        pe = pe * mask.unsqueeze(-1)  # [batch_size, seq_len, d_model]
        
        # 广播机制将位置编码加到输入上
        x = x + pe
        
        return x

class MAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        if self.nhead <= 1:
            # 不使用多头
            self.d_k = d_model
        else:
            assert d_model % nhead == 0
            self.d_k = d_model // nhead
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        
    def forward(self, x, mask=True):
        """
        x: [batch_size, seq_len, d_model]
        mask: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        if self.nhead <= 1:
            # 不使用多头注意力
            Q = self.w_q(x).unsqueeze(1)  # [batch, 1, seq_len, d_model]
            K = self.w_k(x).unsqueeze(1)
            V = self.w_v(x).unsqueeze(1)
        else:
            # 1. 线性投影得到Q/K/V
            Q = self.w_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # [batch, nhead, seq_len, d_k]
            K = self.w_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
            V = self.w_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # 2. 计算缩放点积注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch, nhead, seq_len, seq_len]
        
        # 3. 应用掩码（处理padding + 未来信息遮盖）
        if mask is not None:
            # (1) 生成padding掩码（标记有效数据位置）
            padding_mask = (x.abs().sum(dim=-1) > 0).float()  # [batch_size, seq_len]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]

            # (2) 生成未来时间步掩码（右上半三角掩码，防止信息泄露）
            future_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # [seq_len, seq_len]
            future_mask = future_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            future_mask = future_mask.to(x.device)  # 确保设备一致

            # (3) 合并两种掩码（padding掩码 + 未来掩码）
            combined_mask = (padding_mask == 0) | future_mask  # 任一条件为True时遮盖

            # (4) 应用掩码到注意力分数
            attn_scores = attn_scores.masked_fill(combined_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 4. 加权求和
        output = torch.matmul(attn_weights, V)  # [batch, nhead, seq_len, d_k]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 5. 输出投影
        return self.w_o(output)

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.3)
        
    def forward(self, x):
        return self.linear2(self.LeakyReLU(self.linear1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0):
        super().__init__()
        self.self_attn = MAttention(d_model, nhead)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=True):
        # 自注意力子层
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, d_ff, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=True):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TFDataset(Dataset):
    
    def __init__(self, factor, label):
        self.factor = factor
        self.label = label
        
        # 使用更高效的日期处理
        self.factor['trade_date'] = pd.to_datetime(self.factor['trade_date'], format='%Y%m')
        self.label['trade_date'] = pd.to_datetime(self.label['trade_date'], format='%Y%m')
        self.label['trade_date'] = self.label['trade_date'] - pd.DateOffset(months=1)
        
        # 合并数据
        self.data_origin = pd.merge(self.factor, self.label, on=['ts_code', 'trade_date'], how='inner')
        self.data_origin.dropna(inplace=True)

        # 预计算列名
        feature_cols = [col for col in self.factor.columns if col not in ['ts_code', 'trade_date']]
        label_cols = [col for col in self.label.columns if col not in ['ts_code', 'trade_date']]
        all_data_cols = feature_cols + label_cols
        
        # 使用pivot_table快速重塑数据
        self.data_3d = self._fast_pivot_to_3d(self.data_origin, all_data_cols)
        
        # 如果pivot后还有缺失值，用0填充
        if np.sum(np.isnan(self.data_3d)) > 0:
            self.data_3d = np.nan_to_num(self.data_3d, nan=0.0, posinf=0.0, neginf=0.0)

        # 向量化Winsorize
        self.data_3d = self.winsorize_data_vectorized(self.data_3d, lower=0.1, upper=0.9)
        
        # 转换为tensor
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.data_3d = torch.tensor(self.data_3d, dtype=torch.float32, device=device)
        
        # 存储元信息
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.n_features = len(feature_cols)
        self.n_labels = len(label_cols)
        return

    def _fast_pivot_to_3d(self, data, value_cols):
        """使用pandas pivot快速创建3D数组"""
        stocks = sorted(data['ts_code'].unique())
        timesteps = sorted(data['trade_date'].unique())
        
        # 创建索引映射
        self.stock_to_idx = {stock: i for i, stock in enumerate(stocks)}
        self.time_to_idx = {time: j for j, time in enumerate(timesteps)}
        self.stocks = stocks
        self.timesteps = timesteps
        
        # 预分配3D数组
        n_stocks, n_times, n_features = len(stocks), len(timesteps), len(value_cols)
        data_3d = np.full((n_stocks, n_times, n_features), np.nan, dtype=np.float32)
        
        # 批量处理每个特征 - 向量化操作
        for i, col in enumerate(value_cols):
            try:
                pivot_data = data.pivot_table(
                    index='ts_code', 
                    columns='trade_date', 
                    values=col, 
                    fill_value=np.nan
                )
                pivot_data = pivot_data.reindex(index=stocks, columns=timesteps, fill_value=np.nan)
                data_3d[:, :, i] = pivot_data.values
            except Exception as e:
                print(f"处理特征 {col} 时出错: {e}")
                continue
        
        return data_3d

    def winsorize_data_vectorized(self, data, lower=0.1, upper=0.9):
        """向量化Winsorize处理 - 大幅加速"""
        
        # 重塑为2D进行批量处理
        original_shape = data.shape
        data_2d = data.reshape(-1, data.shape[-1])  # [stocks*timesteps, features]
        
        # 向量化处理每个特征
        for i in range(data.shape[2]):
            feature_data = data_2d[:, i]
            
            # 只对非零值进行Winsorize（保护填充的0值）
            non_zero_mask = feature_data != 0
            
            if np.sum(non_zero_mask) > 0:
                non_zero_data = feature_data[non_zero_mask]
                
                # 向量化计算分位数
                lower_bound = np.quantile(non_zero_data, lower)
                upper_bound = np.quantile(non_zero_data, upper)
                
                # 向量化截尾
                feature_data[non_zero_mask] = np.clip(non_zero_data, lower_bound, upper_bound)
                data_2d[:, i] = feature_data
        
        # 重塑回原始形状
        data = data_2d.reshape(original_shape)
        return data

    def __len__(self):
        return self.data_3d.shape[0]

    def set_time_range(self, start_idx=None, end_idx=None):
        """设置时间范围"""
        if start_idx is not None:
            self.time_start_idx = start_idx
        if end_idx is not None:
            self.time_end_idx = end_idx
    
    def __getitem__(self, idx):
        """根据时间范围返回数据"""
        stock_data = self.data_3d[idx, self.time_start_idx:self.time_end_idx, :]
        
        factor = stock_data[:, :self.n_features]
        label_data = stock_data[:, self.n_features:]
        
        highest_return = label_data[:, 0]
        lowest_return = label_data[:, 1]
        close_return = label_data[:, 2]
        
        return factor, highest_return, lowest_return, close_return


    def get_labels_norm_params(self, train_size):
        """
        计算标签的标准化参数（基于时间维度划分训练集）
        
        参数:
            train_size: 训练集时间比例 (0.0 到 1.0)
        """
        # 按时间维度计算训练集的索引范围
        n_timesteps = self.data_3d.shape[1]
        train_time_idx = int(n_timesteps * train_size)
        
        # 提取所有股票的前train_time_idx个时间步的标签数据
        # [all_stocks, train_timesteps, n_labels]
        train_labels = self.data_3d[:, :train_time_idx, self.n_features:]
        
        label_means = []
        label_stds = []
        
        for i in range(self.n_labels):
            # 提取第i个标签的所有数据
            label_i = train_labels[:, :, i].flatten()  # [all_stocks * train_timesteps]
            
            # 剔除0值
            non_zero_mask = label_i != 0
            if torch.sum(non_zero_mask) > 0:
                non_zero_data = label_i[non_zero_mask]
                mean_i = non_zero_data.mean()
                std_i = non_zero_data.std()
            else:
                print(f"警告：标签 {self.label_cols[i]} 的所有值都为0")
                mean_i = torch.tensor(0.0, device=label_i.device)
                std_i = torch.tensor(1.0, device=label_i.device)
            
            label_means.append(mean_i)
            label_stds.append(std_i)
        
        # 组合结果
        self.label_mean = torch.stack(label_means).unsqueeze(0)  # [1, n_labels]
        self.label_std = torch.stack(label_stds).unsqueeze(0)    # [1, n_labels]
        
        # 避免标准差为0的情况
        self.label_std = torch.clamp(self.label_std, min=1e-8)
        
        # 存储时间划分信息
        self.train_time_idx = train_time_idx
        
        return self.label_mean, self.label_std


class TFDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, train_size=0.5, test_size=0.1):
        """
        TFDataLoader - 专门用于TFDataset的数据加载器
        
        参数:
            dataset: TFDataset实例
            batch_size: 批大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            train_size: 训练集时间比例
            test_size: 测试集时间比例
        """
        self.dataset = dataset
        if train_size + test_size > 1.0:
            raise ValueError("train_size + test_size must be less than 1.0")
        
        self.train_size = train_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 标准化参数
        self.label_mean = None
        self.label_std = None

    def get_norm_params(self, train_size=None):
        """
        获取标准化参数（基于时间维度的训练集）
        """
        if train_size is not None:
            self.train_size = train_size
            
        # 计算标签的标准化参数
        self.dataset.get_labels_norm_params(train_size=self.train_size)
        
        # 保存标准化参数
        self.label_mean = self.dataset.label_mean
        self.label_std = self.dataset.label_std
        
        return self.label_mean, self.label_std

    def get_train_loader(self, train_size=None):
        """
        获取训练集DataLoader（按时间维度分割）
        """
        if train_size is not None:
            self.train_size = train_size
            
        # 计算并应用标准化参数
        self.get_norm_params()
        
        # 设置数据集的时间范围为训练期
        n_timesteps = self.dataset.data_3d.shape[1]
        train_time_idx = int(n_timesteps * self.train_size)
        self.dataset.set_time_range(0, train_time_idx)

        # 对label数据标准化
        self.dataset.data_3d[:, :, self.dataset.n_features:] = (
            self.dataset.data_3d[:, :, self.dataset.n_features:] - self.label_mean
        ) / self.label_std

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers
        )

    def get_test_loader(self, test_size=None):
        """获取测试集loader"""
        if test_size is not None:
            self.test_size = test_size

        # 设置数据集的时间范围为测试期
        n_timesteps = self.dataset.data_3d.shape[1]
        train_time_idx = int(n_timesteps * self.train_size)
        test_time_idx = train_time_idx + int(n_timesteps * self.test_size)
        self.dataset.set_time_range(train_time_idx, test_time_idx)

        # 对label数据标准化
        self.dataset.data_3d[:, :, self.dataset.n_features:] = (
            self.dataset.data_3d[:, :, self.dataset.n_features:] - self.label_mean
        ) / self.label_std

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )
    

class TFLoadData:
    def __init__(self, factor, label, batch_size=32, shuffle=False, num_workers=0, train_size=0.5, test_size=0.1):
        """
        TFLoadData - 数据加载管理类
        
        参数:
            factor: 因子数据DataFrame
            label: 标签数据DataFrame
            batch_size: 批大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            train_size: 训练集时间比例
            test_size: 测试集时间比例
        """
        self.dataset = TFDataset(factor, label)
        self.dataloader = TFDataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            train_size=train_size, 
            test_size=test_size
        )
        
        # 标准化参数
        self.label_mean = None
        self.label_std = None
    
    def get_train_loader(self, train_size=None):
        """
        获取训练集DataLoader
        
        参数:
            train_size: 训练集时间比例
            
        返回:
            train_loader: 训练集DataLoader
        """
        if train_size is not None:
            self.dataloader.train_size = train_size
            
        train_loader = self.dataloader.get_train_loader(train_size=self.dataloader.train_size)
        
        # 保存标准化参数
        self.label_mean = self.dataloader.label_mean
        self.label_std = self.dataloader.label_std
        
        return train_loader

    def get_test_loader(self, test_size=None):
        """
        获取测试集DataLoader
        
        参数:
            test_size: 测试集时间比例
            
        返回:
            test_loader: 测试集DataLoader
        """
        if test_size is not None:
            self.dataloader.test_size = test_size
            
        return self.dataloader.get_test_loader(test_size=self.dataloader.test_size)
    


