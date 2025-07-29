import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1,model_name = "linear_regression"):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.linear.weight, mean=0, std=0.1)
        nn.init.constant_(self.linear.bias, 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.model_name = model_name
        if not os.path.exists("model/checkpoints"):
            os.makedirs("model/checkpoints")
        if not os.path.exists("model/final_models"):
            os.makedirs("model/final_models")
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pth"
        self.model_path = f"model/final_models/{self.model_name}.pth"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        return self.linear(x)
    
    def reset_model(self):
        self.linear.reset_parameters()
        nn.init.normal_(self.linear.weight, mean=0, std=0.1)
        nn.init.constant_(self.linear.bias, 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        return

    
    def train_step(self, train_loader, target = 3,criterion=None):
        if criterion is None:
            criterion = self.loss_fn
        self.train()
        total_loss = 0.0
        for train_data in train_loader:
            inputs = train_data[0].to(self.device)
            targets = train_data[target].to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def fit(self, train_loader, epochs=100, resume_training=False, patience=5):
        start_epoch = 0
        no_improvement_count = 0
        if resume_training:
            start_epoch = self._load_checkpoint()
            print(f"Resuming training from epoch {start_epoch + 1}")

        prev_loss = float('inf')
        for epoch in tqdm(range(start_epoch, epochs), colour='#FA6780'):
            train_loss = self.train_step(train_loader)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")

            # if loss is not improving for 5 epochs, stop training
            if round(train_loss,4) < round(prev_loss,4):
                prev_loss = train_loss
                no_improvement_count = 0    
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    # print("Early stopping triggered due to no improvement in loss.")
                    break
        self.save_model()

    # Predict method & reverse normalization
    def predict(self, test_loader, label_mean, label_std):
        self.eval()
        pred_list = []
        act_list = []
        with torch.no_grad():
            for test_data in test_loader:
                data = test_data[0].to(self.device)
                act_high = test_data[1].to(self.device)
                act_low = test_data[2].to(self.device)
                act_close = test_data[3].to(self.device)
                if act_high.dim() == 1:
                    act_high = act_high.unsqueeze(1)
                if act_low.dim() == 1:
                    act_low = act_low.unsqueeze(1)
                if act_close.dim() == 1:
                    act_close = act_close.unsqueeze(1)
                act = torch.cat([act_high, act_low, act_close], dim=1)
                act = act * label_std + label_mean
                act_list.append(act.cpu().numpy())
                if data.dim() == 1:
                    data = data.unsqueeze(1)
                pred = self(data)
                # reverse normalization
                pred = pred * label_std + label_mean
                pred_list.append(pred.cpu().numpy())
        pred = np.concatenate(pred_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        return pred, act
        
    # nondemeaned R^2 evaluation
    def evaluate(self, test_loader, target = 3):
        self.eval()
        total_sse = 0.0
        total_ss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                inputs = test_data[0].to(self.device)
                targets = test_data[target].to(self.device)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                outputs = self(inputs)
                # Calculate sum of squared errors and total sum of squares
                sse = torch.sum((targets - outputs) ** 2)
                ss = torch.sum(targets ** 2)
                total_sse += sse.item()
                total_ss += ss.item()
        if total_ss < 1e-8:
            return 0
        else:
            return 1 - total_sse / total_ss

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch']
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        torch.save(self.state_dict(), path)
        print (f"Model successfully saved to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        self.load_state_dict(torch.load(path, map_location=self.device))
        print (f"Model successfully loaded from {path}")




class ElasticNet(nn.Module):
    def __init__(self, input_dim, output_dim=1, alpha=1.0, l1_ratio=0.5, model_name = "ElasticNet"):
        super(ElasticNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.linear.weight, mean=0, std=0.1)
        nn.init.constant_(self.linear.bias, 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.model_name = model_name
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if not os.path.exists("model/checkpoints"):
            os.makedirs("model/checkpoints")
        if not os.path.exists("model/final_models"):
            os.makedirs("model/final_models")
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pth"
        self.model_path = f"model/final_models/{self.model_name}.pth"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        return self.linear(x)

    def elastic_net_loss(self, outputs, targets):
        mse_loss = self.loss_fn(outputs, targets)
        l1_reg = torch.sum(torch.abs(self.linear.weight))
        l2_reg = torch.sum(self.linear.weight ** 2)
        elastic_reg = self.alpha * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg)
        return mse_loss + elastic_reg

    def reset_model(self):
        self.linear.reset_parameters()
        nn.init.normal_(self.linear.weight, mean=0, std=0.1)
        nn.init.constant_(self.linear.bias, 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        return

    def train_step(self, train_loader, target = 3,criterion=None):
        if criterion is None:
            criterion = self.elastic_net_loss
        self.train()
        total_loss = 0.0
        for train_data in train_loader:
            inputs = train_data[0].to(self.device)
            targets = train_data[target].to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def fit(self, train_loader, epochs=100, resume_training=False, patience=5):
        start_epoch = 0
        no_improvement_count = 0
        if resume_training:
            start_epoch = self._load_checkpoint()
            print(f"Resuming training from epoch {start_epoch + 1}")

        prev_loss = float('inf')
        for epoch in tqdm(range(start_epoch, epochs), colour='#FA6780'):
            train_loss = self.train_step(train_loader)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")

            # if loss is not improving for 5 epochs, stop training
            if round(train_loss,4) < round(prev_loss,4):
                prev_loss = train_loss
                no_improvement_count = 0    
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    # print("Early stopping triggered due to no improvement in loss.")
                    break
        self.save_model()

    # Predict method & reverse normalization
    def predict(self, test_loader, label_mean, label_std):
        self.eval()
        pred_list = []
        act_list = []
        with torch.no_grad():
            for test_data in test_loader:
                data = test_data[0].to(self.device)
                act_high = test_data[1].to(self.device)
                act_low = test_data[2].to(self.device)
                act_close = test_data[3].to(self.device)
                if act_high.dim() == 1:
                    act_high = act_high.unsqueeze(1)
                if act_low.dim() == 1:
                    act_low = act_low.unsqueeze(1)
                if act_close.dim() == 1:
                    act_close = act_close.unsqueeze(1)
                act = torch.cat([act_high, act_low, act_close], dim=1)
                act = act * label_std + label_mean
                act_list.append(act.cpu().numpy())
                if data.dim() == 1:
                    data = data.unsqueeze(1)
                pred = self(data)
                # reverse normalization
                pred = pred * label_std + label_mean
                pred_list.append(pred.cpu().numpy())
        pred = np.concatenate(pred_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        return pred, act
        
    # nondemeaned R^2 evaluation
    def evaluate(self, test_loader, target = 3):
        self.eval()
        total_sse = 0.0
        total_ss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                inputs = test_data[0].to(self.device)
                targets = test_data[target].to(self.device)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                outputs = self(inputs)
                # Calculate sum of squared errors and total sum of squares
                sse = torch.sum((targets - outputs) ** 2)
                ss = torch.sum(targets ** 2)
                total_sse += sse.item()
                total_ss += ss.item()
        if total_ss < 1e-8:
            return 0
        else:
            return 1 - total_sse / total_ss

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch']
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        torch.save(self.state_dict(), path)
        print (f"Model successfully saved to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        self.load_state_dict(torch.load(path, map_location=self.device))
        print (f"Model successfully loaded from {path}")
    

class NN(nn.Module):
    def __init__(self, input_dim, output_dim=1, alpha=1.0, l1_ratio=0.5, layer = 2, model_name = "NN"):
        super(NN, self).__init__()
        
        if layer == 1:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Tanh()
            )
        elif layer == 2:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, output_dim),
                nn.Tanh()
            )
        elif layer == 3:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, output_dim),
                nn.Tanh()
            )
        elif layer == 4:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, 8),
                nn.LeakyReLU(0.1),
                nn.Linear(8, output_dim),
                nn.Tanh()
            )
        elif layer == 5:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, 8),
                nn.LeakyReLU(0.1),
                nn.Linear(8, 4),
                nn.LeakyReLU(0.1),
                nn.Linear(4, output_dim),
                nn.Tanh()
            )
        else:
            raise ValueError("Unsupported number of layers. Supported values are 1 to 5.")
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.model_name = model_name + f"_{layer}Layers"
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if not os.path.exists("model/checkpoints"):
            os.makedirs("model/checkpoints")
        if not os.path.exists("model/final_models"):
            os.makedirs("model/final_models")
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pth"
        self.model_path = f"model/final_models/{self.model_name}.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.layers(x)

    def elastic_net_loss(self, outputs, targets):
        mse_loss = self.loss_fn(outputs, targets)
        l1_reg = 0
        l2_reg = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                l1_reg += torch.sum(torch.abs(module.weight))
                l2_reg += torch.sum(module.weight ** 2)
        
        elastic_reg = self.alpha * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg)
        return mse_loss + elastic_reg

    def reset_model(self):
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        return

    def train_step(self, train_loader, target = 3,criterion=None):
        if criterion is None:
            criterion = self.loss_fn
        self.train()
        total_loss = 0.0
        for train_data in train_loader:
            inputs = train_data[0].to(self.device)
            targets = train_data[target].to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def fit(self, train_loader, epochs=100, resume_training=False, patience=5):
        start_epoch = 0
        no_improvement_count = 0
        if resume_training:
            start_epoch = self._load_checkpoint()
            print(f"Resuming training from epoch {start_epoch + 1}")

        prev_loss = float('inf')
        for epoch in tqdm(range(start_epoch, epochs), colour='#FA6780'):
            train_loss = self.train_step(train_loader)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}")

            # if loss is not improving for 5 epochs, stop training
            if round(train_loss,4) < round(prev_loss,4):
                prev_loss = train_loss
                no_improvement_count = 0    
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    # print("Early stopping triggered due to no improvement in loss.")
                    break
        self.save_model()

    # Predict method & reverse normalization
    def predict(self, test_loader, label_mean, label_std):
        self.eval()
        pred_list = []
        act_list = []
        with torch.no_grad():
            for test_data in test_loader:
                data = test_data[0].to(self.device)
                act_high = test_data[1].to(self.device)
                act_low = test_data[2].to(self.device)
                act_close = test_data[3].to(self.device)
                if act_high.dim() == 1:
                    act_high = act_high.unsqueeze(1)
                if act_low.dim() == 1:
                    act_low = act_low.unsqueeze(1)
                if act_close.dim() == 1:
                    act_close = act_close.unsqueeze(1)
                act = torch.cat([act_high, act_low, act_close], dim=1)
                act = act * label_std + label_mean
                act_list.append(act.cpu().numpy())
                if data.dim() == 1:
                    data = data.unsqueeze(1)
                pred = self(data)
                # reverse normalization
                pred = pred * label_std + label_mean
                pred_list.append(pred.cpu().numpy())
        pred = np.concatenate(pred_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        return pred, act
        
    # nondemeaned R^2 evaluation
    def evaluate(self, test_loader, target = 3):
        self.eval()
        total_sse = 0.0
        total_ss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                inputs = test_data[0].to(self.device)
                targets = test_data[target].to(self.device)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                outputs = self(inputs)
                # Calculate sum of squared errors and total sum of squares
                sse = torch.sum((targets - outputs) ** 2)
                ss = torch.sum(targets ** 2)
                total_sse += sse.item()
                total_ss += ss.item()
        if total_ss < 1e-8:
            return 0
        else:
            return 1 - total_sse / total_ss

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch']
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        torch.save(self.state_dict(), path)
        print (f"Model successfully saved to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        self.load_state_dict(torch.load(path, map_location=self.device))
        print (f"Model successfully loaded from {path}")


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=5, 
                 min_samples_leaf=2, random_state=42, model_name="RandomForest"):
        from sklearn.ensemble import RandomForestRegressor
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,      # 树的数量
            max_depth=max_depth,            # 树的最大深度
            min_samples_split=min_samples_split,  # 分割节点最小样本数
            min_samples_leaf=min_samples_leaf,    # 叶节点最小样本数
            random_state=random_state,      # 随机种子
            n_jobs=-1                      # 使用所有CPU核心
        )
        
        self.model_name = model_name
        self.feature_names = None
        
        # 创建模型保存目录
        if not os.path.exists("model/checkpoints"):
            os.makedirs("model/checkpoints")
        if not os.path.exists("model/final_models"):
            os.makedirs("model/final_models")
            
        self.checkpoint_path = f"model/checkpoints/{self.model_name}_checkpoint.pkl"
        self.model_path = f"model/final_models/{self.model_name}.pkl"

    def fit(self, train_loader, epochs=None, patience=None):
        """
        训练随机森林模型
        注意：随机森林不需要epochs和patience参数
        """
        X_train, y_train = self._extract_data_from_loader(train_loader)
        self.model.fit(X_train, y_train)
        self.save_model()
        print(f"Random Forest trained with {len(X_train)} samples")

    def predict(self, test_loader, label_mean=None, label_std=None):
        """预测并返回与PyTorch模型相同的格式"""
        X_test, y_test = self._extract_data_from_loader(test_loader)
        pred = self.model.predict(X_test)
        
        # 转换为与PyTorch模型相同的格式
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
            
        return pred, y_test

    def evaluate(self, test_loader, target=3):
        """计算R²分数"""
        from sklearn.metrics import r2_score
        X_test, y_test = self._extract_data_from_loader(test_loader, target)
        pred = self.model.predict(X_test)
        return r2_score(y_test, pred)

    def _extract_data_from_loader(self, data_loader, target=3):
        """从PyTorch DataLoader中提取数据"""
        X_list = []
        y_list = []
        
        for batch in data_loader:
            X_batch = batch[0].numpy()  # 特征
            y_batch = batch[target].numpy()  # 目标
            
            X_list.append(X_batch)
            y_list.append(y_batch)
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        # 确保y是一维数组（sklearn期望的格式）
        if y.ndim > 1:
            y = y.ravel()
            
        return X, y

    def reset_model(self):
        """重置模型（重新初始化）"""
        from sklearn.ensemble import RandomForestRegressor
        
        # 保存原始参数
        params = self.model.get_params()
        self.model = RandomForestRegressor(**params)
        print("Random Forest model reset")

    def save_model(self, path=None):
        """保存模型"""
        import joblib
        if path is None:
            path = self.model_path
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model successfully saved to {path}")

    def load_model(self, path=None):
        """加载模型"""
        import joblib
        if path is None:
            path = self.model_path
            
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Model successfully loaded from {path}")
        else:
            print(f"Model file {path} not found")

    def get_feature_importance(self):
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            print("Model not trained yet")
            return None